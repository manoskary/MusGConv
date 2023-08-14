import os, abc, hashlib
import warnings
import errno
import requests
from git import Repo


def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.
    Codes borrowed from mxnet/gluon/utils.py
    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.
    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash


def makedirs(path):
    try:
        os.makedirs(os.path.expanduser(os.path.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and os.path.isdir(path):
            raise


def makedirs(path):
    try:
        os.makedirs(os.path.expanduser(os.path.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and os.path.isdir(path):
            raise


def extract_archive(file, target_dir, overwrite=False):
    """Extract archive file.

    Parameters
    ----------
    file : str
        Absolute path of the archive file.
    target_dir : str
        Target directory of the archive to be uncompressed.
    overwrite : bool, default True
        Whether to overwrite the contents inside the directory.
        By default always overwrites.
    """
    if os.path.exists(target_dir) and not overwrite:
        return
    print('Extracting file to {}'.format(target_dir))
    if file.endswith('.tar.gz') or file.endswith('.tar') or file.endswith('.tgz'):
        import tarfile
        with tarfile.open(file, 'r') as archive:
            archive.extractall(path=target_dir)
    elif file.endswith('.gz'):
        import gzip
        import shutil
        with gzip.open(file, 'rb') as f_in:
            target_file = os.path.join(target_dir, os.path.basename(file)[:-3])
            with open(target_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    elif file.endswith('.zip'):
        import zipfile
        with zipfile.ZipFile(file, 'r') as archive:
            archive.extractall(path=target_dir)
    else:
        raise Exception('Unrecognized file type: ' + file)


def get_download_dir():
    """Get the absolute path to the download directory.
    Returns
    -------
    dirname : str
        Path to the download directory
    """
    default_dir = os.path.join(os.path.expanduser('~'), '.musgconv')
    dirname = os.environ.get('musgconv_DOWNLOAD_DIR', default_dir)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname


def download(url, path=None, overwrite=True, sha1_hash=None, retries=5, verify_ssl=True, log=True, extract=True):
    """Download a given URL.
    Codes borrowed from mxnet/gluon/utils.py
    Parameters
    ----------
    url : str
        URL to download.
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with the same name as in url.
    overwrite : bool, optional
        Whether to overwrite the destination file if it already exists.
        By default always overwrites the downloaded file.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    retries : integer, default 5
        The number of times to attempt downloading in case of failure or non 200 return codes.
    verify_ssl : bool, default True
        Verify SSL certificates.
    log : bool, default True
        Whether to print the progress for download
    extract : bool, default True
        Whether to extract the downloaded file.
    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split('/')[-1]
        # Empty filenames are invalid
        assert fname, 'Can\'t construct file-name from this URL. ' \
            'Please set the `path` option manually.'
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path
    assert retries >= 0, "Number of retries should be at least 0"

    if not verify_ssl:
        warnings.warn(
            'Unverified HTTPS request is being made (verify_ssl=False). '
            'Adding certificate verification is strongly advised.')

    if overwrite or not os.path.exists(fname) or (sha1_hash and not check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        while retries+1 > 0:
            # Disable pyling too broad Exception
            # pylint: disable=W0703
            try:
                if log:
                    print('Downloading %s from %s...' % (fname, url))
                r = requests.get(url, stream=True, verify=verify_ssl)
                if r.status_code != 200:
                    raise RuntimeError("Failed downloading url %s" % url)
                with open(fname, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                if sha1_hash and not check_sha1(fname, sha1_hash):
                    raise UserWarning('File {} is downloaded but the content hash does not match.'
                                      ' The repo may be outdated or download may be incomplete. '
                                      'If the "repo_url" is overridden, consider switching to '
                                      'the default repo.'.format(fname))
                break
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e
                else:
                    if log:
                        print("download failed, retrying, {} attempt{} left"
                              .format(retries, 's' if retries > 1 else ''))

    if extract:
        extract_archive(fname, os.path.dirname(fname), overwrite=overwrite)

    return


class musgconvDataset(object):
    """The basic musgconv Dataset for creating various datasets.
    This class defines a basic template class for musgconv Dataset.
    The following steps will are executed automatically:

      1. Check whether there is a dataset cache on disk
         (already processed and stored on the disk) by
         invoking ``has_cache()``. If true, goto 5.
      2. Call ``download()`` to download the data.
      3. Call ``process()`` to process the data.
      4. Call ``save()`` to save the processed dataset on disk and goto 6.
      5. Call ``load()`` to load the processed dataset from disk.
      6. Done.

    Users can overwite these functions with their
    own data processing logic.

    Parameters
    ----------
    name : str
        Name of the dataset
    url : str
        Url to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.musgconv/
    save_dir : str
        Directory to save the processed dataset.
        Default: same as raw_dir
    hash_key : tuple
        A tuple of values as the input for the hash function.
        Users can distinguish instances (and their caches on the disk)
        from the same dataset class by comparing the hash values.
        Default: (), the corresponding hash value is ``'f9065fa7'``.
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information

    Attributes
    ----------
    url : str
        The URL to download the dataset
    name : str
        The dataset name
    raw_dir : str
        Raw file directory contains the input data folder
    raw_path : str
        Directory contains the input data files.
        Default : ``os.path.join(self.raw_dir, self.name)``
    save_dir : str
        Directory to save the processed dataset
    save_path : str
        File path to save the processed dataset
    verbose : bool
        Whether to print information
    hash : str
        Hash value for the dataset and the setting.
    """
    def __init__(self, name, features="all", url=None, raw_dir=None, save_dir=None,
                 hash_key=(), force_reload=False, verbose=False):
        self._name = name
        self._url = url
        self._features = features
        self._force_reload = force_reload
        self._verbose = verbose
        self._hash_key = hash_key
        self._hash = self._get_hash()

        # if no dir is provided, the default musgconv download dir is used.
        if raw_dir is None:
            self._raw_dir = get_download_dir()
        else:
            self._raw_dir = raw_dir

        if save_dir is None:
            self._save_dir = self._raw_dir
        else:
            self._save_dir = save_dir

        self._load()

    def download(self):
        """Overwite to realize your own logic of downloading data.

        It is recommended to download the to the :obj:`self.raw_dir`
        folder. Can be ignored if the dataset is
        already in :obj:`self.raw_dir`.
        """
        pass

    def save(self):
        """Overwite to realize your own logic of
        saving the processed dataset into files.

        It is recommended to use ``musgconv.graphs.save()``
        to save musgconv score graph into files and use
        ``musgconv.utils.save_info`` to save extra
        information into files.
        """
        pass

    def load(self):
        """Overwite to realize your own logic of
        loading the saved dataset from files.

        It is recommended to use ``musgconv.utils.load_graph_from_part``
        to load musgconv Score graph from files and use
        ``musgconv.utils.load_info`` to load extra information
        into python dict object.
        """
        pass

    def process(self):
        """Overwrite to realize your own logic of processing the input data.
        """
        raise NotImplementedError

    def has_cache(self):
        """Overwrite to realize your own logic of
        deciding whether there exists a cached dataset.

        By default False.
        """
        return False

    def _download(self):
        """Download dataset by calling ``self.download()`` if the dataset does not exists under ``self.raw_path``.
            By default ``self.raw_path = os.path.join(self.raw_dir, self.name)``
            One can overwrite ``raw_path()`` function to change the path.
        """
        if os.path.exists(self.raw_path):  # pragma: no cover
            return

        makedirs(self.raw_dir)
        self.download()

    def _load(self):
        r"""Entry point from __init__ to load the dataset.
            if the cache exists:
                Load the dataset from saved musgconv score graph and information files.
                If loadin process fails, re-download and process the dataset.
            else:
                1. Download the dataset if needed.
                2. Process the dataset and build the score graph.
                3. Save the processed dataset into files.
        """
        load_flag = not self._force_reload and self.has_cache()

        if load_flag:
            try:
                self.load()
                if self.verbose:
                    print('Done loading data from cached files.')
            except KeyboardInterrupt:
                raise
            except:
                load_flag = False
                print('Loading from cache failed, re-processing.')

        if not load_flag:
            self._download()
            if self.verbose:
                print('Preprocessing data...')
            self.process()
            if self.verbose:
                print('Saving preprocessed data...')
            self.save()
            if self.verbose:
                print('Done saving data into cached files.')

    def _get_hash(self):
        hash_func = hashlib.sha1()
        hash_func.update(str(self._hash_key).encode('utf-8'))
        return hash_func.hexdigest()[:8]

    @property
    def url(self):
        """Get url to download the raw dataset.
        """
        return self._url

    @property
    def name(self):
        r"""Name of the dataset.
        """
        return self._name

    @property
    def raw_dir(self):
        r"""Raw file directory contains the input data folder.
        """
        return self._raw_dir

    @property
    def raw_path(self):
        r"""Directory contains the input data files.
            By default raw_path = os.path.join(self.raw_dir, self.name)
        """
        return os.path.join(self.raw_dir, self.name)

    @property
    def save_dir(self):
        r"""Directory to save the processed dataset.
        """
        return self._save_dir

    @property
    def save_path(self):
        r"""Path to save the processed dataset.
        """
        return os.path.join(self._save_dir, self.name)

    @property
    def verbose(self):
        r"""Whether to print information.
        """
        return self._verbose

    @property
    def hash(self):
        r"""Hash value for the dataset and the setting.
        """
        return self._hash

    @abc.abstractmethod
    def __getitem__(self, idx):
        r"""Gets the data object at index.
        """
        pass

    @abc.abstractmethod
    def __len__(self):
        r"""The number of examples in the dataset."""
        pass


class BuiltinDataset(musgconvDataset):
    """The Basic musgconv Builtin Dataset.

    Parameters
    ----------
    name : str
        Name of the dataset.
    url : str
        Url to download the raw dataset.
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.musgconv/
    hash_key : tuple
        A tuple of values as the input for the hash function.
        Users can distinguish instances (and their caches on the disk)
        from the same dataset class by comparing the hash values.
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: False
    is_zip : bool
    """
    def __init__(self, name, url, raw_dir=None, hash_key=(), force_reload=False, verbose=False, is_zip=False, clone=False, branch=None):
        self.is_zip = is_zip
        self.force_reload = force_reload
        self.clone = clone if not is_zip else False
        if self.clone:
            self.branch = "master" if branch is None else branch
        else:
            self.branch = None
        super(BuiltinDataset, self).__init__(
            name,
            url=url,
            raw_dir=raw_dir,
            save_dir=None,
            hash_key=hash_key,
            force_reload=force_reload,
            verbose=verbose)

    def download(self):
        if self.is_zip or ".zip" in self.url:
            download(self.url, path=os.path.join(self.raw_path, "raw.zip"), extract=True)
        elif "https://github.com/" in self.url or self.clone:
            repo_path = os.path.join(self.raw_dir, self.name)
            Repo.clone_from(self.url, repo_path, single_branch=True, b=self.branch, depth=1)
        else:
            raise ValueError("Unknown url: {}".format(self.url))

