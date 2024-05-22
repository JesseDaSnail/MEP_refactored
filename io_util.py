import tkinter
from tkinter import filedialog
import pickle
from analysis import SimulationResult


def get_file_path():
    root = tkinter.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    file_path = filedialog.askopenfilename()
    root.destroy()
    return file_path


def load_from_file(file_path: str | None = None):
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
