import datetime
import os


def create_unique_directory(top_level_directory: str = './',
                            unique_prefix: str = '',
                            unique_postfix: str = '') -> str:
    """


    Parameters
    ----------
    top_level_directory : str, optional
        Top level directory to create unique directory in. The default is './'.
    unique_prefix : str, optional
        Prefix for unique directory name. The default is ''.
    unique_postfix : str, optional
        Postfix for unique directory. The default is ''.

    Returns
    -------
    str
        Created unique directory with path.  If creation is not successful, '' is returned.

    """
    unique_dir = unique_prefix
    if len(unique_prefix) > 0 and unique_prefix[-1] != '_':
        unique_dir += '_'
    unique_dir += str(datetime.datetime.utcnow().strftime("%Y%m%d_%H-%M-%S"))
    if len(unique_postfix) > 0 and unique_postfix[0] != '_':
        unique_dir += '_'
    unique_dir += unique_postfix
    output_dir_path = top_level_directory
    if len(top_level_directory) > 0 and top_level_directory[-1] != '/':
        output_dir_path += '/'
    output_dir_path = top_level_directory + unique_dir + '/'
    try:
        # Create output directory.
        os.makedirs(output_dir_path, exist_ok=True)
    except:
        output_dir_path = ''
    return output_dir_path
