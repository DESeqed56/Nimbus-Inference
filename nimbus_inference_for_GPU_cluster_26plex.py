# NIMBUS-INFERENCE PREDICTION SCRIPT

# This script will be run on the GPU cluster.


# 0. PREPARATIONS

# import required packages
import warnings
warnings.simplefilter("ignore")
import os
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
from nimbus_inference.nimbus import Nimbus, prep_naming_convention
from nimbus_inference.utils import MultiplexDataset
from alpineer import io_utils
from nimbus_inference import example_dataset
from nimbus_inference.viewer_widget import NimbusViewer

# Set up base directory.
base_dir = os.path.normpath("/processing/r.raukas/base_dir/")


# 1. SET FILE PATHS AND PARAMETERS.

# set up file paths
tiff_dir = os.path.join(base_dir, "image_data")
deepcell_output_dir = os.path.join(base_dir, "segmentation", "deepcell_output")
nimbus_output_dir = os.path.join(base_dir, "nimbus_output")

# Create nimbus output directory
os.makedirs(nimbus_output_dir, exist_ok=True)

# Check if paths exist
io_utils.validate_paths([base_dir, tiff_dir, deepcell_output_dir, nimbus_output_dir])


# 2. SET UP INPUT PATHS AND THE NAMING CONVENTION FOR THE SEGMENTATION DATA.

# define the channels to include
include_channels = [
    "CD14", "CD20", "CD3e", "CD163", "CD31", "CD4", "CD45RO", "EpCAM", "CD66", "CD8", "Pan-Cytokeratin",
    "CD68", "E-cadherin", "SMA", "Collagen IV", "Ki67", "Vimentin", "FOXP3", "CD21", "Caveolin",
    "HLA-DR", "CD39", "PD-1", "IFNG", "PD-L1", "CD56"
]

# either get all fovs in the folder...
fov_names = os.listdir(tiff_dir)
# ... or optionally, select a specific set of fovs manually
# fovs = ["fov0", "fov1"]

# make sure to filter paths out that don't lead to FoVs, e.g. .DS_Store files.
fov_names = [fov_name for fov_name in fov_names if not fov_name.startswith(".")] 

# construct paths for fovs
fov_paths = [os.path.join(tiff_dir, fov_name) for fov_name in fov_names]

# Prepare segmentation naming convention that maps a fov_path to the according segmentation label map
segmentation_naming_convention = prep_naming_convention(deepcell_output_dir)

# test segmentation_naming_convention
if os.path.exists(segmentation_naming_convention(fov_paths[0])):
    print("Segmentation data exists for fov 0 and naming convention is correct")
else:
    print("Segmentation data does not exist for fov 0 or naming convention is incorrect")

# use the `MultiplexDataset` class to abstract away differences in data representation
dataset = MultiplexDataset(
    fov_paths=fov_paths,
    suffix=".ome.tiff", # or .png, .jpg, .jpeg, .tif or .ome.tiff
    include_channels=include_channels,
    segmentation_naming_convention=segmentation_naming_convention,
    output_dir=	nimbus_output_dir,
)


# 3. LOAD MODEL AND INITIALIZE NIMBUS APPLICATION.

# Fallback for setting asset directory if env var is present
nimbus_assets_dir = os.getenv("NIMBUS_ASSETS_DIR")
if nimbus_assets_dir:
    from nimbus_inference import nimbus
    nimbus.ASSETS_DIR = nimbus_assets_dir

# Load Nimbus model.
    # set test_time_aug to 'False' if not using GPU.
    # default batch size is 4.
nimbus = Nimbus(
    dataset=dataset,
    save_predictions=True,
    batch_size=2,
    test_time_aug=True,
    input_shape=[1024,1024],
    device="auto",
    output_dir=nimbus_output_dir,
)

# check if all inputs are valid
nimbus.check_inputs()


# 4. PREPARE NORMALIZATION DIRECTORY.

dataset.prepare_normalization_dict(
    quantile=0.999,
    n_subset=50,
    clip_values=(0, 2),
    multiprocessing=True,
    overwrite=True
)


# 5. MAKE PREDICTIONS WITH THE MODEL.

cell_table = nimbus.predict_fovs()
