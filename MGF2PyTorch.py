import numpy as np
import h5py
import pyteomics.mgf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class MGF2PyTorch:
    """
    A class to parse MGF files and store MS/MS spectra in a compact HDF5 format
    for high-throughput machine learning applications.
    """

    def __init__(self, mgf_file: str = None):
        self.mgf_file = mgf_file

    def parse_mgf(self, mgf_file: str):
        """
        Parses an MGF file and return spectra.
        """
        spectra = []

        with open(mgf_file, 'r') as f:
            spectrum = None
            peaks = []
            for line in f:
                if line.startswith('BEGIN IONS'):
                    spectrum = {}
                    peaks = []
                elif line.startswith('TITLE'):
                    parts = line.split()
                    spectrum['title'] = parts[0].split('=')[1]
                    spectrum['filename'] = parts[1].split('"')[1]
                    spectrum['spectrum_id'] = ' '.join(parts[2:])
                elif line.startswith('RTINSECONDS'):
                    spectrum['rtinseconds'] = float(line.split('=')[1].strip())
                elif line.startswith('PEPMASS'):
                    try:
                        spectrum['pepmass'] = tuple(map(float, line.split('=')[1].split()))
                    except ValueError:
                        spectrum['pepmass'] = line.split('=')[1].strip()
                elif line.startswith('CHARGE'):
                    spectrum['charge'] = int(line.split('=')[1].strip().replace('+', ''))
                elif line.startswith('END IONS'):
                    if spectrum:
                        spectrum["peaks"] = np.array(peaks, dtype=np.float32)
                        spectra.append(spectrum)
                    spectrum = None
                elif ' ' in line:
                    parts = line.split()
                    mz = float(parts[0])
                    intensity = float(parts[1])
                    peaks.append((mz, intensity))

        return spectra

    def read_mgf_pyteomics(self, mgf_file: str):
        """
        Reads an MGF file using pyteomics and returns spectra.
        """
        spectra = []

        with pyteomics.mgf.MGF(mgf_file) as reader:
            for spectrum in reader:
                spectra.append(spectrum)

        return spectra

    def parse_and_store_flat_hdf5(self, mgf_path: str, hdf5_path: str):
        charges = []
        precursors = []
        rts = []
        lengths = []
        mz_all = []
        intensity_all = []

        with open(mgf_path, 'r') as f:
            peaks = []
            spectrum = None
            for line in f:
                if line.startswith('BEGIN IONS'):
                    peaks = []
                    spectrum = {}
                elif line.startswith('TITLE'):
                    parts = line.split()
                    spectrum['title'] = parts[0].split('=')[1]
                    spectrum['filename'] = parts[1].split('"')[1]
                    spectrum['spectrum_id'] = ' '.join(parts[2:])
                elif line.startswith('RTINSECONDS'):
                    spectrum['rt'] = float(line.split('=')[1].strip())
                elif line.startswith('PEPMASS'):
                    try:
                        spectrum['pepmass'] = tuple(map(float, line.split('=')[1].split()))
                    except ValueError:
                        spectrum['pepmass'] = line.split('=')[1].strip()
                elif line.startswith('CHARGE'):
                    spectrum['charge'] = int(line.split('=')[1].replace('+', '').strip())
                elif line.startswith('END IONS'):
                    mz, intensity = zip(*peaks) if peaks else ([], [])
                    mz_all.extend(mz)
                    intensity_all.extend(intensity)
                    lengths.append(len(mz))
                    charges.append(spectrum.get('charge', 0))
                    precursors.append(spectrum.get('pepmass', (0,0))[0])
                    rts.append(spectrum.get('rt', 0.0))
                elif ' ' in line:
                    parts = line.split()
                    mz_val = float(parts[0])
                    intensity_val = float(parts[1])
                    peaks.append((mz_val, intensity_val))

        # Write to HDF5
        with h5py.File(hdf5_path, 'w') as hf:
            hf.create_dataset("mz", data=np.array(mz_all, dtype=np.float32))
            hf.create_dataset("intensity", data=np.array(intensity_all, dtype=np.float32))
            hf.create_dataset("lengths", data=np.array(lengths, dtype=np.int32))
            hf.create_dataset("charges", data=np.array(charges, dtype=np.int8))
            hf.create_dataset("precursors", data=np.array(precursors, dtype=np.float32))
            hf.create_dataset("rts", data=np.array(rts, dtype=np.float32))

    def save_to_hdf5(self, spectra, output_path):
        """
        Saves the parsed spectra data to an HDF5 file.
        """
        with h5py.File(output_path, 'w') as hf:
            for i, spectrum in enumerate(spectra):
                group = hf.create_group(f"spectrum_{i}")
                for key, value in spectrum.items():
                    if key != 'peaks':  # Store metadata
                        group.create_dataset(key, data=value)
                    else:  # Store peaks as a dataset
                        peaks = np.array(spectrum[key])
                        group.create_dataset('peaks', data=peaks)

    def load_from_hdf5(self, hdf5_file):
        """
        Loads spectra data from an HDF5 file.
        """
        spectra = []

        with h5py.File(hdf5_file, 'r') as hf:
            for group_name in hf:
                group = hf[group_name]
                spectrum = {}
                for key in group:
                    data = group[key][()]
                    # Decode bytes to str if needed
                    if isinstance(data, bytes):
                        data = data.decode('utf-8')
                    # Convert 0-d arrays to scalars
                    if isinstance(data, np.ndarray) and data.ndim == 0:
                        data = data.item()
                    spectrum[key] = data
                spectrum['peaks'] = group['peaks'][:]
                spectra.append(spectrum)

        return spectra

def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences in a batch.
    """
    mz = pad_sequence([item['mz'] for item in batch], batch_first=True)
    intensity = pad_sequence([item['intensity'] for item in batch], batch_first=True)
    charge = torch.stack([item['charge'] for item in batch])
    precursor = torch.stack([item['precursor'] for item in batch])
    rt = torch.stack([item['rt'] for item in batch])

    return {
        'mz': mz,
        'intensity': intensity,
        'charge': charge,
        'precursor': precursor,
        'rt': rt
        }

class SpectraDataset(Dataset):
    """
    A PyTorch Dataset class for loading spectra from an HDF5 file.
    """
    def __init__(self, hdf5_path: str):
        self.h5 = h5py.File(hdf5_path, 'r')
        self.mz = self.h5['mz']
        self.intensity = self.h5['intensity']
        self.lengths = self.h5['lengths']
        self.charges = self.h5['charges']
        self.precursors = self.h5['precursors']
        self.rts = self.h5['rts']

        # Build offset array for variable-length spectra.
        self.offsets = np.zeros(len(self.lengths), dtype=np.int64)
        np.cumsum(self.lengths[:-1], out=self.offsets[1:])

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        start = self.offsets[idx]
        end = start + self.lengths[idx]
        mz = self.mz[start:end]
        intensity = self.intensity[start:end]

        return {
            'mz': torch.tensor(mz, dtype=torch.float32),
            'intensity': torch.tensor(intensity, dtype=torch.float32),
            'charge': torch.tensor(self.charges[idx], dtype=torch.int64),
            'precursor': torch.tensor(self.precursors[idx], dtype=torch.float32),
            'rt': torch.tensor(self.rts[idx], dtype=torch.float32)
        }
    def close(self):
        """Closes the HDF5 file."""
        if hasattr(self, 'h5') and self.h5 is not None:
            self.h5.close()
            self.h5 = None

    def __enter__(self):
        if not hasattr(self, 'h5') or self.h5 is None:
            self._open_file()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


# --- Additional Filtering Functions ---
def filter_spectra_by_charge(spectra, charge_value):
    """
    Filters a list of spectra, returning only those with the specified charge value.
    """
    return [s for s in spectra if s.get('charge') == charge_value]

def filter_peaks_by_mz(spectrum, mz_min, mz_max):
    """
    Filters the peaks of a given spectrum to include only those with m/z values within [mz_min, mz_max].
    """
    peaks = spectrum['peaks']
    mask = (peaks[:, 0] >= mz_min) & (peaks[:, 0] <= mz_max)
    return peaks[mask]
