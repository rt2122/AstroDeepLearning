"""Scripts for Planck detector."""
from . import scan_sky_Planck, fast_skan_sky_Planck, sky_extract_catalog
import os


def scan_Planck_Unet(model_path: str, data_path: str, out_path: str, step: str, device: str
                     ) -> None:
    """Full scan for Planck.

    :param model_path: Path to model.
    :type model_path: str
    :param data_path: Path to Planck data in tiles.
    :type data_path: str
    :param out_path: Output directory for scans.
    :type out_path: str
    :param step: Step parameter for scanning.
    :type step: str
    :param device: Device (cpu or gpu).
    :type device: str
    :rtype: None
    """
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if step == "fast":
        print("Fast scan.")
        fast_skan_sky_Planck(data_path, out_path, model_path)
    else:
        print(f"Slow scan with step {step}")
        scan_sky_Planck(data_path, out_path, model_path, int(step))


def extract_cat_Planck(in_path, out_path, thr) -> None:
    """Extract catalog.

    :param in_path: Input path to scans.
    :param out_path: Output path to catalog.
    :param thr: Threshold for masks.
    :rtype: None
    """
    df = sky_extract_catalog(in_path, float(thr))
    df.to_csv(out_path, index=False)
