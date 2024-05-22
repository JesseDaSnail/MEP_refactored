import tkinter
from tkinter import filedialog
import pickle
from analysis import SimulationResult


def get_file_path():
    """
    Get the file path from the user by opening a file dialog.

    This function creates a Tkinter root window, hides it, and sets it as the topmost window.
    It then opens a file dialog using the `filedialog.askopenfilename()` function, which allows
    the user to select a file. The selected file path is returned as the result of the function.

    Returns:
        str: The file path selected by the user.
    """
    root = tkinter.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    file_path = filedialog.askopenfilename()
    root.destroy()
    return file_path


def load_from_file(file_path: str | None = None):
    """
    Load data from a file using pickle.

    This function loads data from a file specified by the `file_path` parameter.
    If `file_path` is not provided, it prompts the user to select a file using a file dialog.
    If the user cancels the file dialog or does not select a file, it prints a message and returns `None`.

    Parameters:
        file_path (str | None): The path of the file to load data from. Defaults to `None`.

    Returns:
        The loaded data from the file, or `None` if no file is selected or the file path is empty.
    """
    if file_path is not None:
        with open(file_path, "rb") as f:
            result = pickle.load(f)
        return result

    file_path = get_file_path()

    if file_path == "":
        print("No file selected. Exiting...")
        return None

    with open(file_path, "rb") as f:
        result = pickle.load(f)

    return result


def save_to_file(result: SimulationResult, file_path: str | None = None):
    """
    Save a SimulationResult object to a file. If no file path is provided, a file dialog is opened to prompt the user to select a file.

    Args:
        result (SimulationResult): The SimulationResult object to save.
        file_path (str | None, optional): The path of the file to save the result to. If None, a file dialog is opened to prompt the user to select a file. Defaults to None.

    Returns:
        None
    """
    if file_path is not None:
        with open(file_path, "wb") as f:
            pickle.dump(result, f)
        return

    if result.sources[0].__class__.__name__ == "ParametricSource":
        filename = (
            f"{result.model.nr*result.model.dr:.1e}m_"
            f"{result.model.nz*result.model.dz:.1e}m_"
            f"{result.model.nt*result.model.dt:.1e}s_"
            f"{result.sources[0].frequency1:.1e}Hz_"
            f"{abs(result.sources[0].frequency1-result.sources[0].frequency2):.1e}Hz_"
            f"{result.sources[0].amplitude:.1e}Pa.pkl"
        )
    else:
        filename = (
            f"{result.model.nr*result.model.dr:.1e}m_"
            f"{result.model.nz*result.model.dz:.1e}m_"
            f"{result.model.nt*result.model.dt:.1e}s_"
            f"{result.sources[0].frequency:.1e}Hz_"
            f"{result.sources[0].amplitude:.1e}Pa.pkl"
        )

    root = tkinter.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    file_path = filedialog.asksaveasfilename(
        initialfile=filename,
        defaultextension=".pkl",
        filetypes=[("Pickle files", "*.pkl")],
    )
    root.destroy()

    if file_path == "":
        print("No file selected. Exiting...")
        return

    with open(file_path, "wb") as f:
        pickle.dump(result, f)
