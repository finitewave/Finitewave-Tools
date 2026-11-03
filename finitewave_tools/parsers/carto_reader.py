from pathlib import Path
from xml.etree import ElementTree
import configparser
import numpy as np
import pandas as pd
import re
from tqdm.auto import tqdm


__all__ = [
    "load_study_file",
    "select_non_empty_maps",
    "load_mesh_file",
    "load_car_file",
    "load_signals",
    "load_egms",
    "load_xml_egms",
    "load_raw_signal",
    "load_default_signal",
    "extract_anatomical_tags",
    "load_xml_signals",
]


def load_study_file(path, filename):
    """Load CARTO study xml file.

    Parameters:
    -----------
        path : str or Path
            Path to the directory with files.
        filename : str
            Name of the study xml file.

    Returns:
    -------
        ElementTree: CARTO XML map data.
    """
    file = Path(path).joinpath(filename).with_suffix(".xml")

    if not file.is_file():
        raise FileNotFoundError(f"File not found {file}")

    root = ElementTree.parse(file).getroot()
    maps = root.findall("Maps")
    return maps

def select_non_empty_maps(xml_data):
    """Get maps from xml data which contain any point.

    Parameters:
    -----------
        xml_data : ElementTree
            CARTO XML map data.

    Returns:
    -------
        dict: All maps which contain at least one point.
    """
    out = {}
    for map_name in xml_data.findall("Map"):
        points = map_name.find("CartoPoints")
        if int(points.attrib["Count"]) == 0:
            continue
        out[map_name.attrib["Name"]] = map_name
    return out


def load_car_file(path, map_name, pid=2, x=4, y=5, z=6, uv=10, bv=11, lat=12, **kwargs):
    """Load CARTO car file.

    Parameters:
    -----------
        path : str or Path
            Path to the directory with files.
        map_name : str or XML Element
            File name or XML Element containing the
            map data.

    Returns:
    -------
        pd.DataFrame: CARTO points with avaliable values.
    """
    if isinstance(map_name, ElementTree.Element):
        map_name = map_name.attrib["Name"]

    columns_dict = {"PID" : pid, "x" : x, "y" : y, "z" : z, "Unipolar" : uv,
                    "Bipolar" : bv, "LAT" : lat}

    for key, value in kwargs.items():
        columns_dict[key] = value

    path = Path(path).joinpath(map_name + "_car").with_suffix(".txt")
    if not path.is_file():
        raise FileNotFoundError(f"File not found {path.stem}")
    
    if columns_dict is None:
        data = np.genfromtxt(path, skip_header=1)
        return pd.DataFrame(data)

    data = np.genfromtxt(path, usecols=columns_dict.values(), skip_header=1)
    data = pd.DataFrame(data, columns=columns_dict.keys(), dtype=float)
    data["PID"] = data["PID"].astype(int)
    return data


def load_mesh_file(path, map_name):
    """Load CARTO mesh file.

    Parameters:
    -----------
        path : str or Path
            Path to the directory with files.
        map_name : str or XML Element
            File name or XML Element containing the
            map data.

    Returns:
    -------
        tuple(pd.DataFrame, pd.DataFrame): Vertices and triangles of the mesh
            with avaliable values.
    """
    def parse_mesh_file(parser, section, columns, n):
        array = np.genfromtxt([parser[section][str(i)] for i in range(n)])
        return pd.DataFrame(array, columns=columns)

    if isinstance(map_name, ElementTree.Element):
        map_name = map_name.attrib["Name"]

    path = Path(path).joinpath(map_name).with_suffix(".mesh")

    if not path.is_file():
        raise FileNotFoundError(f"File not found {path}")

    parser = configparser.ConfigParser()
    parser.read(path, encoding="latin")
    num_of_vertices = int(parser["GeneralAttributes"]["numvertex"])
    num_of_triangles = int(parser["GeneralAttributes"]["numtriangle"])

    # Mesh nodes
    if "VerticesSection" in parser.sections():
        columns = ["x", "y", "z", "nx", "ny", "nz", "group_id"]
        vertices = parse_mesh_file(
            parser, "VerticesSection", columns, num_of_vertices
        )
    else:
        raise KeyError("VerticesSection not found")

    # Mesh triangles
    if "TrianglesSection" in parser.sections():
        columns = ["v0", "v1", "v2", "nx", "ny", "nz", "group_id"]
        triangles = parse_mesh_file(
            parser, "TrianglesSection", columns, num_of_triangles
        )
        triangles["v0"] = triangles["v0"].astype(int)
        triangles["v1"] = triangles["v1"].astype(int)
        triangles["v2"] = triangles["v2"].astype(int)
    else:
        raise KeyError("TrianglesSection not found")

    # Mesh colors
    if "VerticesColorsSection" in parser.sections():
        columns = parser["GeneralAttributes"]["colorsnames"].split()
        colors = parse_mesh_file(
            parser, "VerticesColorsSection", columns, num_of_vertices
        )

        vertices = vertices.join(colors)

    return vertices, triangles


def extract_anatomical_tags(map_):
    """
    Get anatomical tags from study map.

    Args:
        map (XML Element): 'Map' Element from Study XML file

    Return:
        list: List of coordinates for anatomical tags
    """
    out = []
    anatomical_tags = map_.find('Anatomical_Tags')

    if anatomical_tags is None:
        return []

    anatomical_tags = anatomical_tags.findall('Anatomical_Tag')
    for anatomical_tag in anatomical_tags:
        line_loop = anatomical_tag.find('Perimiter').find('LINE_LOOP')
        count = int(line_loop.attrib['Count'])

        if count == 0:
            continue

        coords = np.array([float(x) for x in line_loop.text.split()]).reshape((count, 3))
        out.append(coords)

    return out



def load_signals(
    path,
    map_,
    points,
    gain=1,
    channels=None,
):
    """Load electrograms associated with points from a '_car.txt' file.

    Args:
        path (str): Path to the directory containing data.
        map_ (str or XML Element): File name or XML Element containing the
            map data.
        points (list): List of points to load data for.
        gain (float, optional): Raw ECG to mV conversion coefficient.
            Defaults to 1.
        channels (dict, optional): Dictionary of channel names and
            corresponding column names.

    Returns:
        out (list): Dataframes with signal data.
    """
    if channels is None:
        channels = {
            "Unipolar": "M1",
            "Bipolar": "M1-M2",
            "CS": "CS1-CS2",
            "ECG": "I",
        }

    if isinstance(map, ElementTree.Element):
        map_ = map_.attrib["Name"]

    out = []
    for point in points:
        pattern = f"_P{point}_ECG_Export.txt"
        out.append(load_signal(path, map_, point, gain, channels, pattern))
    out = pd.concat(out, ignore_index=True)

    out = out.groupby("Point_ID").agg(
        dict(zip(channels.keys(), len(channels) * [list]))
    )

    return out.reset_index()


def load_xml_egms(path, map_name, pids=None):
    file_data = load_xml_signals(path, map_name)

    if pids is not None:
        file_data = file_data[file_data["PID"].isin(pids)]

    collected_egms = {}

    for _, row in tqdm(file_data.iterrows(), total=len(file_data), desc="Loading ECGs"):
        path_egm = Path(path).joinpath(row["filename"])
        if not path_egm.is_file():
            print(f"Skipping {path_egm.stem}")
            continue

        default_channels = {}
        for channel in ["Unipolar", "Bipolar", "Reference"]:
            default_channels[channel] = row[channel]

        pid = row["PID"]

        signal = load_default_signal(path_egm, default_channels)
        collected_egms[pid] = signal

    return collected_egms


def load_egms(path, map_name, pids=None):
    """Load electrograms from multiple export files.

    Parameters:
    -----------
        path : str or Path
            Path to the directory with files.
        map_name : str or XML Element
            File name or XML Element containing the map data.

    Returns:
    -------
        tuple : int, pd.DataFrame
            List of point IDs and DataFrame with signal data.
    """
    egm_files = [p for p in list(Path(path).glob(f"{map_name}_P*_ECG_Export.txt"))]

    out = {}
    for egm_file in egm_files:
        if not egm_file.is_file():
            print(f"Skipping {egm_file.stem}")
            continue

        pid = int(egm_file.stem.split("_P")[-1].split("_ECG")[0])

        if pids is not None and pid not in pids:
            continue

        res = load_default_signal(egm_file)
        out[pid] = res
    return out


def load_default_signal(filepath, channels=None):
    """Load default electrograms from single export file.

    Parameters:
    -----------
        filepath : str or Path
            Path to the file containing signal data.

    Returns:
        pd.DataFrame
            DataFrame with signal data.
    """
    gain, headers, default_channels, columns = load_signal_headers(filepath)
    columns_idx = np.arange(len(columns))

    if channels is not None:
        default_channels.update(channels)

    if len(default_channels) < 1:
        raise ValueError("No default channels found in the file header.")

    bi_channel = default_channels["Bipolar"]

    if len(bi_channel.split("-")) > 1:
        left_channel, right_channel = bi_channel.split("-")

        if len(left_channel) > len(right_channel):
            right_channel = left_channel.split("_")[0] + "_" + right_channel
        
        if left_channel not in default_channels.values():
            default_channels["Unipolar2"] = left_channel

        if right_channel not in default_channels.values():
            default_channels["Unipolar2"] = right_channel

    columns_idx = [columns.index(channel) for channel in default_channels.values()]
    columns = default_channels.values()

    data = gain * np.genfromtxt(filepath, skip_header=headers, usecols=columns_idx)
    data = pd.DataFrame(data, columns=columns)

    return data


def load_raw_signal(filepath):
    """Load electrograms from single export file.

    Parameters:
    -----------
        filepath : str or Path
            Path to the file containing signal data.

    Returns:
        pd.DataFrame
            DataFrame with signal data.
    """
    gain, headers, _, columns = load_signal_headers(filepath)

    columns_idx = np.arange(len(columns))

    data = gain * np.genfromtxt(filepath, skip_header=headers, usecols=columns_idx)
    data = pd.DataFrame(data, columns=columns)

    return data


def are_all_numbers(line):
    tokens = line.split()
    only_numbers = bool(tokens) and all(
        re.fullmatch(r"-?\d+(?:\.\d+)?", x) for x in tokens
    )
    return only_numbers


def load_signal_headers(filepath, max_headers=10):
    filepath = Path(filepath)

    if not filepath.is_file():
        raise FileNotFoundError(f"File not found {filepath.stem}")
    
    default_channels = {}
    gain = 1.0
    headers = 0

    with open(filepath, "r", encoding="latin") as f:
        line = f.readline()
        headers = 1
        prev_line = line

        for _ in range(1, max_headers):
            prev_line = line
            line = f.readline().strip()

            if are_all_numbers(line):
                break

            if "Raw ECG to MV (gain)" in line:
                gain = float(update_metadata("Raw ECG to MV (gain)", line))

            if "Reference Channel" in line:
                default_channels["Reference"] = update_metadata(
                    "Reference Channel", line
                )

            if "Unipolar Mapping Channel" in line:
                default_channels["Unipolar"] = update_metadata(
                    "Unipolar Mapping Channel", line
                )

            if "Bipolar Mapping Channel" in line:
                default_channels["Bipolar"] = update_metadata(
                    "Bipolar Mapping Channel", line
                )

            headers += 1
            if are_all_numbers(line):
                break

        else:
            raise ValueError(f"Too many headers in {filepath.stem}. Check search pattern for data headers.")

    columns = [re.sub(r"\(.*?\)", "", column) for column in prev_line.strip().split()]
    return gain, headers, default_channels, columns



def update_metadata(pattern, line):
    line = "=".join([a.strip() for a in line.split("=")])
    return line.split(pattern + "=")[1].split()[0]


def load_point_export_list(path, map_name):
    return list(path.glob(f"{map_name}_P*_Point_Export*.xml"))


def load_point_export_xml(data_xml):
    root = ElementTree.parse(data_xml).getroot()

    point_id = root.attrib["ID"]
    egm_xml = root.find("ECG")
    egm_file = egm_xml.attrib["FileName"]
    uni_channel = egm_xml.attrib["UnipolarMappingChannel"]
    bipolar_channel = egm_xml.attrib["BipolarMappingChannel"]
    ref_channel = egm_xml.attrib["ReferenceChannel"]

    res = {
        "PID": int(point_id),
        "filename": egm_file,
        "Unipolar": uni_channel,
        "Bipolar": bipolar_channel,
        "Reference": ref_channel,
    }
    return res


def load_xml_signals(path, map_name):
    point_exports = load_point_export_list(path, map_name)
    out = []
    for point_export in point_exports:
        out.append(load_point_export_xml(point_export))

    return pd.DataFrame(out)
