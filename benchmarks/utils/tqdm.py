
def _is_notebook():
    try:
        shell = get_ipython().__class__
        return shell.__name__ == "ZMQInteractiveShell" or "colab" in shell.__module__
    except NameError:
        return False

if _is_notebook():
    from tqdm.notebook import tqdm, trange
else:
    from tqdm import tqdm, trange


