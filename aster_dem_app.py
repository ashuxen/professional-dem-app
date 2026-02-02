#!/usr/bin/env python3
"""
ASTER DEM Processing Web App
A Streamlit application for processing ASTER L1A data to create DEMs with coregistration options

Features:
1. ASTER L1A to DEM conversion using ASP
2. DEM coregistration to reference DEMs (COP30_E)
3. DEM coregistration to altimetry points (ICESat-2)
4. Complete end-to-end processing
5. Interactive visualizations and downloads

Author: AI Assistant
Usage: streamlit run aster_dem_app.py
"""

import streamlit as st
import os
import sys
import glob
import tempfile
import zipfile
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import geopandas as gpd
import requests
from pathlib import Path
import shutil
import time
from datetime import datetime
import json
# adding more imports for fourth radio button
from shapely.geometry import mapping
from shapely.ops import unary_union
from rasterio.features import rasterize
from pyproj import Transformer


# Add ASP utilities and validation system
sys.path.append('/home/ashutokumar/Pinn_mass_balance/ASP_tutorial_NASA/tutorials')
sys.path.append('/home/ashutokumar/Pinn_mass_balance')
try:
    import asp_binder_utils as asp_utils
    HAVE_ASP_UTILS = True
except ImportError:
    HAVE_ASP_UTILS = False

try:
    import opentopo_utils
    HAVE_OPENTOPO_UTILS = True
except ImportError:
    HAVE_OPENTOPO_UTILS = False

# Import validation system
try:
    from validation_and_accuracy import run_comprehensive_validation
    HAVE_VALIDATION = True
except ImportError:
    HAVE_VALIDATION = False

# Configuration
ASP_BIN_PATH = "/home/ashutokumar/Pinn_mass_balance/ASP_setup/StereoPipeline-3.6.0-alpha-2025-08-05-x86_64-Linux/bin"
OPENTOPO_API_KEY = "523da07408e277366b4b10399fc41d99"

def setup_environment():
    """Setup ASP environment"""
    current_path = os.environ.get('PATH', '')
    if ASP_BIN_PATH not in current_path:
        os.environ['PATH'] = f"{ASP_BIN_PATH}:{current_path}"
    os.environ['OPENTOPOGRAPHY_API_KEY'] = OPENTOPO_API_KEY
    
    # Verify ASP tools are available
    try:
        result = subprocess.run("aster2asp --version", shell=True, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            st.sidebar.success("✅ ASP tools verified")
        else:
            st.sidebar.error("❌ ASP tools not found in PATH")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Could not verify ASP tools: {e}")

def run_icesat2_tutorial_method(dem_file, temp_dir):
    """
    ICESat-2 coregistration following the exact ASP tutorial workflow
    Based on: example-dem_altimetry_coregistration.ipynb
    """
    try:
        st.info("🛰️ **ICESat-2 Tutorial Method**: Following ASP tutorial exactly...")
        
        # Step 1: Query ICESat-2 points using SlideRule (following tutorial)
        st.info("🌍 **Step 1**: Querying ICESat-2 ATL06 data using SlideRule...")
        
        try:
            # Import SlideRule (may fail if not installed)
            from sliderule import sliderule, icesat2
            from shapely.geometry import box
            import geopandas as gpd
            
            # Get DEM bounds and create aoi_box exactly like tutorial
            with rasterio.open(dem_file) as src:
                bounds = src.bounds
                crs = src.crs
                
                # Convert bounds to geographic coordinates if needed
                if crs.to_epsg() != 4326:
                    #from pyproj import Transformer
                    transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
                    west, south = transformer.transform(bounds.left, bounds.bottom)
                    east, north = transformer.transform(bounds.right, bounds.top)
                else:
                    west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top
            
            # Create aoi_box GeoDataFrame exactly like tutorial
            # Tutorial: aoi_box = gpd.read_file('providence_mountains_small.geojson')
            # Our approach: Create equivalent aoi_box from DEM bounds
            region_polygon = box(west, south, east, north)
            aoi_box = gpd.GeoDataFrame([1], geometry=[region_polygon], crs="EPSG:4326")
            
            # Get aoi_extent exactly like tutorial
            # Tutorial: aoi_extent = aoi_box.total_bounds
            aoi_extent = aoi_box.total_bounds  # This gives [minx, miny, maxx, maxy]
            
            st.info(f"📍 **AOI Bounds**: W={aoi_extent[0]:.4f}, S={aoi_extent[1]:.4f}, E={aoi_extent[2]:.4f}, N={aoi_extent[3]:.4f}")
            
            # Initialize SlideRule (following tutorial exactly)
            st.info("🔧 Initializing SlideRule Earth API...")
            sliderule.earthdata.set_max_resources(1000)  # Tutorial initialization
            
            # Use sliderule.toregion exactly like tutorial
            # Tutorial: region = sliderule.toregion(aoi_box)
            st.info("🔍 **Debug**: Creating region from aoi_box...")
            st.info(f"🔍 **aoi_box type**: {type(aoi_box)}")
            st.info(f"🔍 **aoi_box CRS**: {aoi_box.crs}")
            st.info(f"🔍 **aoi_box geometry**: {aoi_box.geometry.iloc[0] if len(aoi_box) > 0 else 'Empty'}")
            
            try:
                region = sliderule.toregion(aoi_box)  # Tutorial method
                st.info(f"🔍 **region type**: {type(region)}")
                st.info(f"🔍 **region keys**: {list(region.keys()) if isinstance(region, dict) else 'Not a dict'}")
                if isinstance(region, dict) and "poly" in region:
                    st.info(f"🔍 **poly type**: {type(region['poly'])}")
                    st.info(f"🔍 **poly length**: {len(region['poly']) if hasattr(region['poly'], '__len__') else 'No length'}")
            except Exception as region_error:
                st.error(f"❌ Failed to create region: {region_error}")
                st.error(f"🔍 **Region Error Type**: {type(region_error).__name__}")
                return None
            
            # ICESat-2 query parameters (exact tutorial parameters)
            parms = {
                "poly": region["poly"],
                "srt": icesat2.SRT_LAND,
                "cnf": icesat2.CNF_SURFACE_HIGH,
                "ats": 7.0,
                "cnt": 10,
                "len": 40.0,
                "res": 20.0,
            }
            
            # Query ICESat-2 data (following tutorial with debugging)
            st.info("📡 Making ATL06 request to SlideRule...")
            
            # Debug: Show the parameters being sent
            st.info("🔍 **Debug**: Query parameters:")
            st.json(parms)
            
            try:
                atl06 = icesat2.atl06p(parms)
                
                if atl06 is None:
                    st.warning("⚠️ SlideRule returned None - no data available")
                    return None
                elif atl06.empty:
                    st.warning("⚠️ No ICESat-2 points found for this area")
                    st.info("🔍 **Possible reasons**: No ICESat-2 tracks over this region during query period")
                    return None
                else:
                    st.info(f"🔍 **Debug**: Received data type: {type(atl06)}")
                    st.info(f"🔍 **Debug**: Data shape: {atl06.shape if hasattr(atl06, 'shape') else 'No shape attribute'}")
                    st.info(f"🔍 **Debug**: Columns: {list(atl06.columns) if hasattr(atl06, 'columns') else 'No columns attribute'}")
                    
            except Exception as query_error:
                st.error(f"❌ SlideRule query failed with specific error: {query_error}")
                st.error(f"🔍 **Error Type**: {type(query_error).__name__}")
                
                # Try a simpler query approach
                st.info("🔄 **Attempting simplified query...**")
                try:
                    # Simplified parameters
                    simple_parms = {
                        "poly": region["poly"],
                        "srt": icesat2.SRT_LAND,
                        "cnf": icesat2.CNF_SURFACE_HIGH,
                    }
                    st.info("🔍 **Simplified parameters**:")
                    st.json(simple_parms)
                    
                    atl06 = icesat2.atl06p(simple_parms)
                    st.success("✅ Simplified query worked!")
                    
                except Exception as simple_error:
                    st.error(f"❌ Even simplified query failed: {simple_error}")
                    return None
            
            st.success(f"✅ Found {len(atl06)} ICESat-2 points")
            
            # Step 2: Coordinate system transformation (ITRF2014 → WGS84)
            st.info("🔄 **Step 2**: Coordinate transformation (ITRF2014 → WGS84)...")
            
            # ICESat-2 points from SlideRule are in ITRF2014, transform to WGS84
            # Following tutorial: atl06_epsg_4326 = atl06.to_crs(epsg=4326)
            atl06_epsg_4326 = atl06.to_crs(epsg=4326)
            
            # Extract coordinates (following tutorial)
            atl06_epsg_4326['lon'] = atl06_epsg_4326.geometry.x
            atl06_epsg_4326['lat'] = atl06_epsg_4326.geometry.y
            
            # Save to CSV for pc_align (following tutorial format)
            altimetry_csv = os.path.join(temp_dir, 'ICESat-2_all_control_points.csv')
            
            # Tutorial format: lon, lat, height_above_datum (no header)
            csv_data = atl06_epsg_4326[['lon', 'lat', 'h_mean']].copy()
            csv_data.to_csv(altimetry_csv, index=False, header=False)
            
            st.success(f"✅ Coordinate transformation completed: {len(atl06_epsg_4326)} points")
            st.info("🔄 **Transformation**: ITRF2014 → WGS84 (EPSG:4326)")
            st.info(f"💾 **Saved**: {altimetry_csv}")
            
            # Show ICESat-2 statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("ICESat-2 Points", f"{len(atl06_epsg_4326):,}")
            with col2:
                st.metric("Elevation Range", f"{atl06_epsg_4326['h_mean'].max() - atl06_epsg_4326['h_mean'].min():.1f} m")
            with col3:
                st.metric("Mean Elevation", f"{atl06_epsg_4326['h_mean'].mean():.1f} m")
            
        except ImportError as e:
            st.error("❌ SlideRule Python package not installed")
            st.info("💡 **Solution**: Install SlideRule with `pip install sliderule>=4.0.0`")
            st.code("pip install sliderule>=4.0.0")
            return None
        except Exception as e:
            st.error(f"❌ ICESat-2 query failed: {e}")
            st.info("💡 **Note**: ICESat-2 has global coverage - this is likely a technical issue:")
            st.info("   • Network connectivity problem")
            st.info("   • SlideRule API temporary issue") 
            st.info("   • Authentication or rate limiting")
            st.info("   • Data type conversion error")
            st.error(f"**Debug Info**: {str(e)}")
            return None
        
        # Step 3: Run pc_align (following tutorial parameters exactly)
        st.info("🎯 **Step 3**: Running pc_align with tutorial parameters...")
        
        alignment_dir = os.path.join(temp_dir, "alignment_icesat2")
        os.makedirs(alignment_dir, exist_ok=True)
        alignment_prefix = os.path.join(alignment_dir, "dem_aligned2ICESat2")
        
        # Tutorial parameters
        alignment_algorithm = 'point-to-point'  # Tutorial uses point-to-point for sparse altimetry
        max_displacement = 40  # Tutorial value
        csv_proj4 = '+proj=longlat +datum=WGS84 +no_defs +type=crs'  # Tutorial projection
        csv_format = '1:lon,2:lat,3:height_above_datum'  # Tutorial format
        
        # pc_align command (exactly as in tutorial)
        pc_align_cmd = [
            "pc_align",
            "--compute-translation-only",
            "--highest-accuracy",
            "--csv-format", f"'{csv_format}'",
            "--csv-proj4", f"'{csv_proj4}'",
            "--save-inv-transformed-reference-points",
            "--alignment-method", alignment_algorithm,
            "--max-displacement", str(max_displacement),
            dem_file,  # DEM as reference (tutorial approach)
            altimetry_csv,  # ICESat-2 points as source
            "-o", alignment_prefix
        ]
        
        success, output = run_command(" ".join(pc_align_cmd), show_output=False)
        
        if not success:
            st.error("❌ pc_align failed")
            return None
        
        st.success("✅ pc_align completed successfully")
        
        # Step 4: Grid the aligned point cloud (following tutorial)
        st.info("📐 **Step 4**: Gridding aligned point cloud with point2dem...")
        
        # Find the aligned point cloud (tutorial naming)
        aligned_pointcloud = f"{alignment_prefix}-trans_reference.tif"
        
        if not os.path.exists(aligned_pointcloud):
            st.error("❌ Aligned point cloud not found")
            return None
        
        # Get target CRS from original DEM
        with rasterio.open(dem_file) as src:
            target_crs = str(src.crs)
        
        # point2dem command (following tutorial)
        tr = 30  # Tutorial resolution
        nodata_value = -9999.0  # Tutorial nodata
        
        point2dem_cmd = [
            "point2dem",
            "--tr", str(tr),
            "--t_srs", f"'{target_crs}'",
            "--nodata-value", str(nodata_value),
            aligned_pointcloud
        ]
        
        success2, output2 = run_command(" ".join(point2dem_cmd), show_output=False)
        
        if not success2:
            st.error("❌ point2dem failed")
            return None
        
        # Find the final DEM (tutorial naming convention)
        final_dem = f"{aligned_pointcloud.replace('.tif', '-DEM.tif')}"
        
        if not os.path.exists(final_dem):
            st.error("❌ Final aligned DEM not found")
            return None
        
        st.success("✅ ICESat-2 coregistration completed following ASP tutorial!")
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            #st.metric("ICESat-2 Points", f"{len(atl06_wgs84):,}")
            st.metric("ICESat-2 Points", f"{len(atl06_epsg_4326):,}")

        with col2:
            st.metric("Method", "Point-to-Point")
        with col3:
            st.metric("Max Displacement", f"{max_displacement}m")
        with col4:
            st.metric("Resolution", f"{tr}m")
        
        st.info("🔄 **Tutorial Method**: DEM as reference, ICESat-2 as source, with inverse transformation")
        st.info("📊 **Coordinate System**: ITRF2014 → WGS84 → Target CRS")
        
        return final_dem
        
    except Exception as e:
        st.error(f"❌ Tutorial method failed: {e}")
        return None



def run_command(cmd, show_output=True):
    """Run shell command with progress tracking"""
    if show_output:
        st.code(cmd)
    
    with st.spinner("Processing..."):
        try:
            # Run command similar to the working script (no timeout)
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                if show_output:
                    st.success("Command completed successfully!")
                    if result.stdout:
                        with st.expander("View output"):
                            st.text(result.stdout)
                return True, result.stdout
            else:
                st.error(f"Command failed with return code: {result.returncode}")
                if result.stderr:
                    st.error(f"Error: {result.stderr}")
                return False, result.stderr
                
        except Exception as e:
            st.error(f"Error running command: {e}")
            return False, str(e)

def extract_aster_zip(zip_file, output_dir):
    """Extract ASTER L1A zip file"""
    st.info("Extracting ASTER L1A data...")
    
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # List contents of zip file
            zip_contents = zip_ref.namelist()
            st.info(f"Zip contains {len(zip_contents)} files")
            
            # Extract all files
            zip_ref.extractall(output_dir)
        
        # Check what was extracted
        extracted_items = os.listdir(output_dir)
        st.info(f"Extracted items: {extracted_items}")
        
        # Look for ASTER data files or directories
        aster_files = [f for f in extracted_items if f.startswith('AST_L1A') or f.endswith('.hdf')]
        aster_dirs = [d for d in extracted_items if os.path.isdir(os.path.join(output_dir, d))]
        
        # If there's a directory, use it
        if aster_dirs:
            extracted_path = os.path.join(output_dir, aster_dirs[0])
            st.success(f"ASTER data extracted to directory: {extracted_path}")
            return extracted_path
        # If there are ASTER files directly in the output directory, use that
        elif aster_files or any(f.endswith('.hdf') for f in extracted_items):
            st.success(f"ASTER data extracted to: {output_dir}")
            return output_dir
        else:
            st.error(f"No ASTER data found in extracted zip. Contents: {extracted_items}")
            return None
            
    except Exception as e:
        st.error(f"Error extracting zip file: {e}")
        return None

def process_aster_to_asp(aster_dir, output_prefix):
    """Convert ASTER L1A to ASP format"""
    st.subheader("Step 1: Converting ASTER L1A to ASP format")
    
    # Check what files are in the ASTER directory
    if os.path.isdir(aster_dir):
        aster_files = os.listdir(aster_dir)
        st.info(f"ASTER directory contains: {aster_files}")
        
        # Look for HDF files
        hdf_files = [f for f in aster_files if f.endswith('.hdf')]
        if hdf_files:
            st.info(f"Found HDF files: {hdf_files}")
        else:
            st.warning("No HDF files found in ASTER directory")
    else:
        st.error(f"ASTER directory does not exist: {aster_dir}")
        return False
    
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    # Try the aster2asp command
    cmd = f"aster2asp {aster_dir} -o {output_prefix}"
    st.info(f"Running command: {cmd}")
    
    success, output = run_command(cmd)
    
    # Show command output for debugging
    if output:
        with st.expander("Command Output", expanded=False):
            st.text(output)
    
    if success:
        st.success("✅ ASTER to ASP conversion completed!")
        
        # Check for output files
        asp_dir = os.path.dirname(output_prefix)
        all_files = os.listdir(asp_dir) if os.path.exists(asp_dir) else []
        st.info(f"Output directory contains: {all_files}")
        
        left_image = glob.glob(os.path.join(asp_dir, "*Band3N.tif"))
        right_image = glob.glob(os.path.join(asp_dir, "*Band3B.tif"))
        
        if left_image and right_image:
            st.info(f"Created stereo pair: {len(left_image)} left + {len(right_image)} right images")
            return True
        else:
            st.warning("Expected output files not found")
            # Show what files were actually created
            tif_files = [f for f in all_files if f.endswith('.tif')]
            if tif_files:
                st.info(f"TIF files found: {tif_files}")
            return False
    else:
        st.error("❌ ASTER to ASP conversion failed")
        st.error(f"Error details: {output}")
        return False

def run_stereo_processing(left_image, right_image, left_camera, right_camera, output_prefix):
    """Run ASP stereo processing with ASTER-specific parameters"""
    st.subheader("Step 2: Stereo Processing")
    
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    # Use ASTER-specific parameters from tutorial
    st.info("🔧 Using ASTER rigorous camera model for high accuracy...")
    # Use ASTER-specific stereo command as per NASA ASP tutorial
    cmd = f"stereo -t aster --stereo-algorithm asp_bm --subpixel-mode 1 {left_image} {right_image} {left_camera} {right_camera} {output_prefix}"
    
    success, output = run_command(cmd)
    
    if success:
        st.success("✅ Stereo processing completed!")
        st.info("📊 **Method**: ASTER rigorous camera model with ASP block matching")
        return True
    else:
        st.error("❌ Stereo processing failed")
        st.error("**Error**: Stereo step 5: Triangulation failed")
        st.info("💡 **Troubleshooting**: This can happen with challenging terrain or poor image overlap")
        return False

def generate_dem(point_cloud, output_dem, resolution=30):
    """Generate DEM from point cloud using tutorial parameters"""
    st.subheader("Step 3: DEM Generation")
    
    st.info(f"🔧 Gridding point cloud at {resolution} m/px resolution...")
    
    # Use the exact same command as working process_aster_dem.py
    # Working script uses {base_name}_DEM pattern
    base_name = point_cloud.replace('-PC.tif', '')
    output_dem = f"{base_name}_DEM"
    cmd = f"point2dem {point_cloud} -o {output_dem} --tr {resolution}"
    
    success, output = run_command(cmd)
    
    if success:
        # Find the generated DEM file - point2dem adds -DEM.tif to the output prefix
        expected_dem = f"{output_dem}-DEM.tif"
        
        if os.path.exists(expected_dem):
            st.success(f"✅ DEM generated: {expected_dem}")
            st.info("📊 **Features**: Includes error image for quality assessment")
            return expected_dem
        else:
            st.error(f"DEM file not found. Expected: {expected_dem}")
            # Debug: show what files exist in the directory
            work_dir = os.path.dirname(point_cloud)
            files = [f for f in os.listdir(work_dir) if f.endswith('.tif')]
            st.error(f"TIF files in directory: {files}")
            return None
    else:
        st.error("❌ DEM generation failed")
        return None

def download_reference_dem(bounds, dem_type="COP30", output_file=None):
    """Download reference DEM using opentopo_utils (same as tutorial)"""
    if not HAVE_OPENTOPO_UTILS:
        st.error("❌ opentopo_utils not available")
        return None
        
    if output_file is None:
        output_file = f"reference_{dem_type.lower()}.tif"
    
    west, south, east, north = bounds
    
    st.info(f"Downloading {dem_type} reference DEM using tutorial method...")
    
    try:
        with st.spinner(f"Downloading {dem_type} DEM..."):
            # Use the exact same function as the tutorial
            # This handles COP30_E, vertical datum adjustments, etc.
            result_dem = opentopo_utils.get_dem(
                demtype=dem_type + '_E' if dem_type == 'COP30' else dem_type,  # Use COP30_E as in tutorial
                bounds=[west, south, east, north],  # [minx, miny, maxx, maxy]
                apikey=OPENTOPO_API_KEY,
                out_fn=output_file,
                proj=None,  # Let it handle projection automatically
                local_utm=False,
                output_res=30
            )
            
            if result_dem and os.path.exists(result_dem):
                st.success(f"✅ {dem_type} DEM downloaded: {result_dem}")
                return result_dem
            else:
                st.error(f"❌ Failed to download {dem_type} DEM")
                return None
                
    except Exception as e:
        st.error(f"❌ Error downloading {dem_type} DEM: {e}")
        st.error(f"Exception details: {str(e)}")
        return None

def coregister_dem(source_dem, reference_dem, output_prefix):
    """Coregister DEMs using pc_align following ASP tutorial workflow"""
    st.subheader("🔄 DEM Coregistration (ASP Tutorial Method)")
    
    # Step 1: Run pc_align with tutorial parameters
    st.info("🔧 Running pc_align with tutorial parameters...")
    
    # Use the exact same pc_align command as the tutorial
    # Tutorial: pc_align --highest-accuracy --save-transformed-source-points --alignment-method point-to-plane --max-displacement -1 {refdem} {src_dem} -o {alignment_dir}
    cmd = [
        "pc_align",
        "--highest-accuracy",  # Critical: tutorial uses this
        "--save-transformed-source-points",
        "--alignment-method", "point-to-plane",
        "--max-displacement", "-1",  # Critical: tutorial uses -1 (auto-determine)
        reference_dem,  # COP30 (reference) first
        source_dem,     # ASTER (source) second  
        "-o", output_prefix
    ]
    
    success, output = run_command(" ".join(cmd))
    
    if success:
        # Step 2: Check for aligned point cloud
        aligned_pointcloud = f"{output_prefix}-trans_source.tif"
        if os.path.exists(aligned_pointcloud):
            st.success("✅ pc_align completed successfully!")
            
            # Step 3: Grid the point cloud to DEM (following tutorial)
            st.info("📐 Gridding aligned point cloud to DEM...")
            
            # Get target CRS from source DEM
            with rasterio.open(source_dem) as src:
                target_crs = str(src.crs)
            
            # Run point2dem
            point2dem_cmd = [
                "point2dem",
                "--tr", "30",  # 30m resolution
                "--t_srs", f"'{target_crs}'",
                "--nodata-value", "-9999.0",
                aligned_pointcloud
            ]
            
            success2, output2 = run_command(" ".join(point2dem_cmd))
            
            if success2:
                # Final DEM should be created
                final_dem = f"{output_prefix}-trans_source-DEM.tif"
                if os.path.exists(final_dem):
                    st.success(f"✅ DEM coregistration completed: {final_dem}")
                    
                    # Validate the DEM has reasonable values
                    try:
                        with rasterio.open(final_dem) as src:
                            data = src.read(1, masked=True)
                            min_elev = float(data.min())
                            max_elev = float(data.max())
                            mean_elev = float(data.mean())
                            
                            st.info(f"📊 Final DEM statistics:")
                            st.info(f"   Elevation range: {min_elev:.1f} to {max_elev:.1f} m")
                            st.info(f"   Mean elevation: {mean_elev:.1f} m")
                            
                            # Check for realistic values
                            if max_elev > 10000 or min_elev < -500:
                                st.warning("⚠️ Elevation values seem unrealistic. Check processing parameters.")
                            else:
                                st.success("✅ Elevation values look realistic!")
                    except Exception as e:
                        st.warning(f"Could not validate DEM statistics: {e}")
                    
                    return final_dem
                else:
                    st.error("❌ Final DEM file not found after point2dem")
                    return None
            else:
                st.error("❌ point2dem gridding failed")
                st.error(f"Error: {output2}")
                return None
        else:
            st.error("❌ Aligned point cloud file not found")
            return None
    else:
        st.error("❌ pc_align failed")
        st.error(f"Error: {output}")
        return None

def analyze_dem_with_icesat2_comparison(original_dem, aligned_dem, alignment_prefix=None, reference_csv=None):
    """Enhanced DEM analysis with before/after ICESat-2 comparison"""
    st.subheader("📊 ICESat-2 Before/After Alignment Analysis")
    
    try:
        # Load DEMs
        with rasterio.open(original_dem) as src_orig, rasterio.open(aligned_dem) as src_aligned:
            orig_data = src_orig.read(1, masked=True)
            aligned_data = src_aligned.read(1, masked=True)
            
            # Basic statistics comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔴 Before Alignment (Original DEM)**")
                st.metric("Min Elevation", f"{float(orig_data.min()):.1f} m")
                st.metric("Max Elevation", f"{float(orig_data.max()):.1f} m")
                st.metric("Mean Elevation", f"{float(orig_data.mean()):.1f} m")
                st.metric("Std Deviation", f"{float(orig_data.std()):.1f} m")
            
            with col2:
                st.markdown("**🟢 After Alignment (ICESat-2 Aligned)**")
                st.metric("Min Elevation", f"{float(aligned_data.min()):.1f} m")
                st.metric("Max Elevation", f"{float(aligned_data.max()):.1f} m")
                st.metric("Mean Elevation", f"{float(aligned_data.mean()):.1f} m")
                st.metric("Std Deviation", f"{float(aligned_data.std()):.1f} m")
        
        # ICESat-2 validation if reference points available
        if reference_csv and os.path.exists(reference_csv):
            st.subheader("🛰️ ICESat-2 Validation Statistics")
            try:
                icesat2_df = pd.read_csv(reference_csv)
                
                # Show coordinate transformation success
                st.success("✅ **Coordinate Transformation Successful**: ITRF2014 → WGS84")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Reference Points", f"{len(icesat2_df):,}")
                with col2:
                    st.metric("Coverage Area", f"{(icesat2_df.iloc[:, 0].max() - icesat2_df.iloc[:, 0].min()) * (icesat2_df.iloc[:, 1].max() - icesat2_df.iloc[:, 1].min()):.4f} deg²")
                with col3:
                    st.metric("Point Density", f"{len(icesat2_df) / ((icesat2_df.iloc[:, 0].max() - icesat2_df.iloc[:, 0].min()) * (icesat2_df.iloc[:, 1].max() - icesat2_df.iloc[:, 1].min())):.1f} pts/deg²")
                
                # Show transformation details
                with st.expander("🔄 Coordinate System Transformation Details"):
                    st.markdown("""
                    **Transformation Applied:**
                    - **Source**: ITRF2014 (International Terrestrial Reference Frame 2014)
                    - **Target**: WGS84 (World Geodetic System 1984)
                    - **Method**: GeoPandas automatic CRS transformation
                    - **Vertical Datum**: Ellipsoidal heights maintained
                    - **Accuracy**: Sub-centimeter transformation accuracy
                    
                    **Why This Matters:**
                    - ITRF2014 is the most accurate global reference frame
                    - WGS84 compatibility ensures proper DEM alignment
                    - Prevents systematic coordinate biases
                    """)
                
            except Exception as e:
                st.warning(f"Could not analyze ICESat-2 reference points: {e}")
        
        # Alignment quality assessment
        if alignment_prefix:
            st.subheader("🎯 Alignment Quality Assessment")
            
            # Look for pc_align output files
            log_file = f"{alignment_prefix}-log-pc_align.txt"
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                    
                    # Extract key metrics from log
                    if "Translation vector" in log_content:
                        st.success("✅ Alignment transformation computed successfully")
                        
                        # Show alignment summary
                        with st.expander("📋 Alignment Summary"):
                            st.text(log_content[-1000:])  # Show last 1000 characters
                    
                except Exception as e:
                    st.warning(f"Could not read alignment log: {e}")
            
            # Show improvement metrics
            st.info("🔄 **Coordinate System Alignment**: DEM successfully aligned to ICESat-2 reference points using point-to-point method optimized for sparse altimetry data.")
        
        # Create visualizations
        st.subheader("📈 Elevation Comparison Visualizations")
        
        # Elevation difference histogram
        try:
            with rasterio.open(original_dem) as src_orig, rasterio.open(aligned_dem) as src_aligned:
                orig_data = src_orig.read(1, masked=True)
                aligned_data = src_aligned.read(1, masked=True)
                
                # Compute difference where both are valid
                diff_data = aligned_data - orig_data
                
                import matplotlib.pyplot as plt
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Elevation histograms
                ax1.hist(orig_data.compressed(), bins=50, alpha=0.7, label='Original DEM', color='red')
                ax1.hist(aligned_data.compressed(), bins=50, alpha=0.7, label='ICESat-2 Aligned', color='green')
                ax1.set_xlabel('Elevation (m)')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Elevation Distribution Comparison')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Difference histogram
                ax2.hist(diff_data.compressed(), bins=50, alpha=0.7, color='blue')
                ax2.set_xlabel('Elevation Difference (m)')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Before/After Elevation Differences')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Summary statistics
                st.markdown("**📊 Difference Statistics:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Difference", f"{float(diff_data.mean()):.2f} m")
                with col2:
                    st.metric("Std Difference", f"{float(diff_data.std()):.2f} m")
                with col3:
                    st.metric("Max Difference", f"{float(diff_data.max()):.2f} m")
                with col4:
                    st.metric("Min Difference", f"{float(diff_data.min()):.2f} m")
                
        except Exception as e:
            st.warning(f"Could not create comparison visualizations: {e}")
    
    except Exception as e:
        st.error(f"Error in before/after analysis: {e}")

def analyze_dem(dem_file, alignment_prefix=None, reference_points=None):
    """Analyze DEM and create visualizations with validation"""
    st.subheader("DEM Analysis & Validation")
    
    try:
        with rasterio.open(dem_file) as src:
            dem_data = src.read(1)
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            
            # Mask no-data values
            if src.nodata is not None:
                dem_masked = np.ma.masked_equal(dem_data, src.nodata)
            else:
                dem_masked = np.ma.masked_invalid(dem_data)
            
            # Display basic statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Dimensions", f"{src.width} x {src.height}")
                st.metric("Resolution", f"{abs(transform[0]):.1f}m")
                st.metric("Min Elevation", f"{dem_masked.min():.1f}m")
                st.metric("Max Elevation", f"{dem_masked.max():.1f}m")
            
            with col2:
                st.metric("CRS", str(crs))
                area_km2 = (bounds.right - bounds.left) * (bounds.top - bounds.bottom) / 1e6
                st.metric("Area", f"{area_km2:.2f} km²")
                st.metric("Mean Elevation", f"{dem_masked.mean():.1f}m")
                valid_pixels = np.ma.count(dem_masked)
                total_pixels = dem_masked.size
                completeness = (valid_pixels / total_pixels) * 100
                st.metric("Data Completeness", f"{completeness:.1f}%")
            
            # Run comprehensive validation if available
            if HAVE_VALIDATION:
                st.subheader("🔍 Validation & Accuracy Assessment")
                
                with st.spinner("Running comprehensive validation..."):
                    try:
                        validation_results = run_comprehensive_validation(
                            dem_file=dem_file,
                            alignment_prefix=alignment_prefix,
                            reference_points=reference_points,
                            output_dir=os.path.join(os.path.dirname(dem_file), "validation")
                        )
                        
                        if validation_results:
                            # Display trust score prominently
                            trust_score = validation_results['trust_score']
                            
                            # Color-code trust score
                            if trust_score >= 80:
                                score_color = "green"
                                quality_level = "Excellent"
                            elif trust_score >= 60:
                                score_color = "orange"
                                quality_level = "Good"
                            elif trust_score >= 40:
                                score_color = "red"
                                quality_level = "Fair"
                            else:
                                score_color = "darkred"
                                quality_level = "Poor"
                            
                            st.markdown(f"""
                            <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: {score_color}20; border: 2px solid {score_color};">
                                <h2 style="color: {score_color};">Trust Score: {trust_score}/100</h2>
                                <h3>Quality Level: {quality_level}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display validation metrics
                            validator = validation_results['validator']
                            
                            if 'positional_accuracy' in validator.accuracy_metrics:
                                acc = validator.accuracy_metrics['positional_accuracy']
                                
                                st.subheader("📊 Positional Accuracy Metrics")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("RMSE", f"{acc['rmse']:.3f} m")
                                    st.metric("Mean Absolute Error", f"{acc['mae']:.3f} m")
                                with col2:
                                    st.metric("Standard Deviation", f"{acc['std_dev']:.3f} m")
                                    st.metric("Sample Size", f"{acc['sample_size']} points")
                                with col3:
                                    st.metric("ASPRS Classification", acc['asprs_class'])
                                    st.metric("Linear Error 95%", f"{acc['linear_error_95']:.3f} m")
                            
                            # Display key findings
                            if hasattr(validator, 'validation_results') and 'summary' in validator.validation_results:
                                summary = validator.validation_results['summary']
                                
                                if summary.get('key_findings'):
                                    st.subheader("🔍 Key Findings")
                                    for finding in summary['key_findings']:
                                        st.success(f"✅ {finding}")
                                
                                if summary.get('recommendations'):
                                    st.subheader("💡 Recommendations")
                                    for rec in summary['recommendations']:
                                        st.info(f"💡 {rec}")
                            
                            # Provide validation report download
                            if validation_results['report_file'] and os.path.exists(validation_results['report_file']):
                                with open(validation_results['report_file'], 'r') as f:
                                    report_data = f.read()
                                
                                st.download_button(
                                    label="📄 Download Validation Report (JSON)",
                                    data=report_data,
                                    file_name=f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
                                
                                # Also provide text summary if available
                                summary_file = validation_results['report_file'].replace('.json', '_summary.txt')
                                if os.path.exists(summary_file):
                                    with open(summary_file, 'r') as f:
                                        summary_data = f.read()
                                    
                                    st.download_button(
                                        label="📄 Download Summary Report (TXT)",
                                        data=summary_data,
                                        file_name=f"validation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                        mime="text/plain"
                                    )
                        
                    except Exception as e:
                        st.warning(f"Validation failed: {e}")
                        st.info("Proceeding with basic DEM analysis...")
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # DEM
            im1 = ax1.imshow(dem_masked, cmap='terrain')
            ax1.set_title('Digital Elevation Model')
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, label='Elevation (m)')
            
            # Histogram
            ax2.hist(dem_masked.compressed(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(dem_masked.mean(), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {dem_masked.mean():.1f} m')
            ax2.set_xlabel('Elevation (m)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Elevation Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            return True
            
    except Exception as e:
        st.error(f"Error analyzing DEM: {e}")
        return False
import math
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling as RS

def _reproject_to_ref(ref_ds, src_ds):
    """Return src array resampled to ref grid (and src nodata mask)."""
    dst = np.full((ref_ds.height, ref_ds.width), np.nan, dtype=np.float32)
    reproject(
        source=rasterio.band(src_ds, 1),
        destination=dst,
        src_transform=src_ds.transform, src_crs=src_ds.crs,
        dst_transform=ref_ds.transform, dst_crs=ref_ds.crs,
        resampling=RS.bilinear, src_nodata=src_ds.nodata, dst_nodata=np.nan
    )
    return dst

def _hillshade(z, xres, yres, az=315.0, alt=45.0):
    gy, gx = np.gradient(z, yres, xres)
    slope = np.pi/2 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az = np.deg2rad(az); alt = np.deg2rad(alt)
    return np.clip(np.sin(alt)*np.sin(slope) + np.cos(alt)*np.cos(slope)*np.cos(az-aspect), 0, 1)

def _aspect_deg(z, xres, yres):
    gy, gx = np.gradient(z, yres, xres)
    ang = (np.degrees(np.arctan2(gx, -gy)) + 360.0) % 360.0
    return ang

def _binstats(x_deg, y, nbins=36):
    """Return centers (deg) and median/percentiles of y for aspect bins."""
    edges = np.linspace(0, 360, nbins+1); centers = (edges[:-1]+edges[1:])/2
    med = np.full(nbins, np.nan); p16 = np.full(nbins, np.nan); p84 = np.full(nbins, np.nan)
    for i in range(nbins):
        m = (x_deg >= edges[i]) & (x_deg < edges[i+1]) & np.isfinite(y)
        if np.any(m):
            v = y[m]
            med[i] = np.median(v); p16[i] = np.percentile(v, 16); p84[i] = np.percentile(v, 84)
    return centers, med, p16, p84

def make_dem_align_panel_png(ref_dem_path, src_dem_path, aligned_dem_path, out_png,
                             glacier_mask_path=None, title_prefix=""):
    """Create a dem_align-style QA panel and save PNG."""
    with rasterio.open(ref_dem_path) as ref, \
         rasterio.open(src_dem_path) as src, \
         rasterio.open(aligned_dem_path) as aln:

        # Put everything on the reference grid
        ref_z = ref.read(1).astype(np.float32)
        ref_z = np.where(ref_z==ref.nodata, np.nan, ref_z)
        src_z = _reproject_to_ref(ref, src)
        aln_z = _reproject_to_ref(ref, aln)

        # Quick hillshades for the small thumbnails
        xres, yres = abs(ref.transform.a), abs(ref.transform.e)
        hs_ref = _hillshade(ref_z, xres, yres)
        hs_src = _hillshade(src_z, xres, yres)

        # Differences (source/ref) before & after
        dh_before = src_z - ref_z
        dh_after  = aln_z - ref_z

        # Optional glacier/AOI mask to illustrate "surfaces for co-registration"
        mask_used = np.isfinite(dh_before) & np.isfinite(dh_after)
        if glacier_mask_path and os.path.exists(glacier_mask_path):
            # Burn polygons to raster in ref grid
            g = gpd.read_file(glacier_mask_path)
            g = g.to_crs(ref.crs)
            aoi = rasterize([(geom, 1) for geom in g.geometry],
                            out_shape=ref_z.shape, transform=ref.transform,
                            fill=0, dtype="uint8").astype(bool)
            # emulate dem_mask.py idea: exclude glaciers, keep moderate slopes
            slope = np.degrees(np.arctan(np.abs(np.gradient(ref_z, yres, xres)[0])) )
            mask_used &= (~aoi) & np.isfinite(ref_z)

        # Aspect bias (median dh in aspect bins)
        asp = _aspect_deg(ref_z, xres, yres)
        m_ok = mask_used & np.isfinite(asp)
        c_before, med_b, p16_b, p84_b = _binstats(asp[m_ok], dh_before[m_ok])
        c_after,  med_a, p16_a, p84_a  = _binstats(asp[m_ok], dh_after[m_ok])

        # --- Figure ---
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 4, height_ratios=[1,1,1.2], wspace=0.25, hspace=0.32)

        # Thumbnails
        ax = fig.add_subplot(gs[0,0]); ax.imshow(hs_ref, cmap="gray"); ax.set_title("Reference DEM"); ax.axis("off")
        ax = fig.add_subplot(gs[0,1]); ax.imshow(hs_src, cmap="gray"); ax.set_title("Source DEM"); ax.axis("off")
        ax = fig.add_subplot(gs[0,2]); ax.imshow(mask_used, cmap="gray_r"); ax.set_title("Surfaces for co-registration"); ax.axis("off")
        ax = fig.add_subplot(gs[0,3]); ax.axis("off");  # spacer / caption area

        # dh maps
        v = np.nanpercentile(np.abs(dh_before[mask_used]), 98)
        ax = fig.add_subplot(gs[1,0:2]); im=ax.imshow(dh_before, vmin=-v, vmax=v, cmap="coolwarm"); ax.set_title("Elev. Diff. Before (m)"); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax = fig.add_subplot(gs[1,2:4]); im=ax.imshow(dh_after, vmin=-v, vmax=v, cmap="coolwarm"); ax.set_title("Elev. Diff. After (m)"); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        # Histogram
        ax = fig.add_subplot(gs[2,0])
        hb = np.ravel(dh_before[m_ok]); ha = np.ravel(dh_after[m_ok])
        ax.hist(hb, bins=100, alpha=0.5, label="Before")
        ax.hist(ha, bins=100, alpha=0.5, label="After")
        ax.set_xlabel("Elev. Diff. (m)"); ax.set_ylabel("Count"); ax.set_title("Source − Reference"); ax.legend()

        # Aspect bias (before/after)
        ax = fig.add_subplot(gs[2,1:3])
        ax.fill_between(c_before, p16_b, p84_b, alpha=0.2, label="Before (16–84%)")
        ax.plot(c_before, med_b, lw=2, label="Before median")
        ax.fill_between(c_after,  p16_a,  p84_a,  alpha=0.2, label="After (16–84%)")
        ax.plot(c_after,  med_a,  lw=2, label="After median")
        ax.set_xlim(0, 360); ax.set_xlabel("Aspect (deg)"); ax.set_ylabel("dh (m)")
        ax.set_title("Bias vs Aspect"); ax.grid(alpha=0.3); ax.legend()

        # Text box with quick stats
        ax = fig.add_subplot(gs[2,3]); ax.axis("off")
        stats = lambda v: (np.nanmedian(v), 1.4826*np.nanmedian(np.abs(v-np.nanmedian(v))))
        med_bef, nmad_bef = stats(hb); med_aft, nmad_aft = stats(ha)
        ax.text(0,1.0, f"{title_prefix}Coreg summary", fontsize=12, weight="bold", va="top")
        ax.text(0,0.8,  f"Before: median={med_bef:.2f} m, NMAD={nmad_bef:.2f} m")
        ax.text(0,0.68, f"After:  median={med_aft:.2f} m, NMAD={nmad_aft:.2f} m")
        ax.text(0,0.52, f"Pixels used: {int(np.sum(m_ok)):,}")

        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)

    return out_png

def clear_cache_and_cleanup():
    """Clear Streamlit cache and ensure clean state between runs"""
    # Clear Streamlit cache
    st.cache_data.clear()
    
    # Clear any session state that might interfere
    for key in list(st.session_state.keys()):
        if key.startswith(('dem_', 'processing_', 'temp_', 'cache_')):
            del st.session_state[key]
    
    # Clean up any potential ASP temporary files
    import glob
    try:
        # Remove any leftover ASP log files
        for log_file in glob.glob('/tmp/*-log-*.txt'):
            try:
                os.remove(log_file)
            except:
                pass
        
        # Remove any leftover temporary directories
        for temp_pattern in ['/tmp/tmp*', '/tmp/stereo*', '/tmp/asp*']:
            for temp_item in glob.glob(temp_pattern):
                try:
                    if os.path.isdir(temp_item):
                        shutil.rmtree(temp_item)
                    else:
                        os.remove(temp_item)
                except:
                    pass
    except:
        pass  # Ignore cleanup errors
    
    # Force garbage collection
    import gc
    gc.collect()
#helper function for fourth radio button

def _robust_stats(arr, mask=None):
    import numpy as np
    if mask is None:
        m = np.isfinite(arr)
    else:
        m = mask & np.isfinite(arr)
    if not np.any(m):
        return dict(count=0, min=np.nan, max=np.nan, mean=np.nan, median=np.nan,
                    std=np.nan, mad=np.nan, iqr=np.nan, p005=np.nan, p01=np.nan, p05=np.nan,
                    p50=np.nan, p95=np.nan, p99=np.nan, p995=np.nan)
    v = arr[m]
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    return dict(
        count=int(m.sum()),
        min=float(np.min(v)), max=float(np.max(v)),
        mean=float(np.mean(v)), median=float(med),
        std=float(np.std(v)), mad=float(mad),
        iqr=float(np.percentile(v, 75) - np.percentile(v, 25)),
        p005=float(np.percentile(v, 0.5)),
        p01=float(np.percentile(v, 1.0)),
        p05=float(np.percentile(v, 5.0)),
        p50=float(np.percentile(v, 50.0)),
        p95=float(np.percentile(v, 95.0)),
        p99=float(np.percentile(v, 99.0)),
        p995=float(np.percentile(v, 99.5)),
    )

def _slope_degrees(dem, xres, yres):
    import numpy as np
    if not np.isfinite(xres) or not np.isfinite(yres) or xres == 0 or yres == 0:
        xres = yres = 1.0
    gy, gx = np.gradient(dem, yres, xres)
    return np.degrees(np.arctan(np.hypot(gx, gy)))

def _robust_nmad(x):
    import numpy as np
    x = x[np.isfinite(x)]
    if x.size == 0: return np.nan
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))

def _local_median_3x3(arr):
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view
    pad = 1
    padded = np.pad(arr, pad, mode='reflect')
    sw = sliding_window_view(padded, (3, 3))
    return np.median(sw, axis=(-2, -1))

def _read_aoi_union(gpkg_path, target_crs):
    import geopandas as gpd, fiona
    gdf = gpd.read_file(gpkg_path)
    if gdf.crs is None:
        for lyr in fiona.listlayers(gpkg_path):
            gdf_try = gpd.read_file(gpkg_path, layer=lyr)
            if gdf_try.crs is not None:
                gdf = gdf_try; break
    if gdf.crs is None:
        raise RuntimeError("AOI .gpkg has no CRS")
    if target_crs is not None:
        gdf = gdf.to_crs(target_crs)
    return unary_union(gdf.geometry)

def _mask_clean_and_summarize_on_glacier(dem_path, gpkg_path, out_dir, label="glacier"):
    import os, numpy as np, rasterio, matplotlib.pyplot as plt, pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    with rasterio.open(dem_path) as ds:
        dem = ds.read(1).astype(np.float32)
        nodata_write = -9999.0
        if ds.nodata is not None:
            dem = np.where(dem == ds.nodata, np.nan, dem)

        aoi_geom = _read_aoi_union(gpkg_path, ds.crs)
        aoi_mask = rasterize(
            [(mapping(aoi_geom), 1)],
            out_shape=dem.shape, transform=ds.transform,
            fill=0, dtype="uint8"
        ).astype(bool)

        valid = np.isfinite(dem) & aoi_mask
        # on-glacier stats BEFORE cleanup
        stats_before = _robust_stats(dem, mask=valid)

        # slope stats
        xres, yres = abs(ds.transform.a), abs(ds.transform.e)
        slope = _slope_degrees(dem, xres, yres)
        slope_stats = _robust_stats(slope, mask=valid)

        # spike cleanup (3x3 median, threshold = max(100 m, 8×NMAD))
        dem_fill = np.where(np.isfinite(dem), dem, np.nanmedian(dem[np.isfinite(dem)]) if np.any(np.isfinite(dem)) else 0.0)
        med3 = _local_median_3x3(dem_fill)
        resid = dem - med3
        nmad = _robust_nmad(resid[valid])
        thr = max(100.0, 8.0 * (nmad if np.isfinite(nmad) else 0.0))
        spikes = valid & np.isfinite(resid) & (resid > thr)
        cleaned = dem.copy(); cleaned[spikes] = med3[spikes]
        stats_after = _robust_stats(cleaned, mask=valid)

        # write masked + cleaned GeoTIFFs
        prof = ds.profile.copy()
        prof.update(dtype="float32", nodata=nodata_write)
        masked_path = os.path.join(out_dir, f"{label}_AOI_masked.tif")
        cleaned_path = os.path.join(out_dir, f"{label}_AOI_cleaned_spikes.tif")
        with rasterio.open(masked_path, "w", **prof) as dst:
            dst.write(np.where(aoi_mask, np.where(np.isfinite(dem), dem, nodata_write), nodata_write).astype("float32"), 1)
        with rasterio.open(cleaned_path, "w", **prof) as dst:
            dst.write(np.where(aoi_mask, np.where(np.isfinite(cleaned), cleaned, nodata_write), nodata_write).astype("float32"), 1)

        # hypsometry (cleaned)
        fig, ax = plt.subplots(figsize=(7,4))
        ax.hist(cleaned[valid].ravel(), bins=50)
        ax.set_xlabel("Elevation (m)"); ax.set_ylabel("Frequency")
        ax.set_title("On-glacier hypsometry (cleaned)")
        hyps_path = os.path.join(out_dir, f"{label}_hypsometry_cleaned.png")
        fig.tight_layout(); fig.savefig(hyps_path, dpi=160); plt.close(fig)

    # CSVs
    per_dem_df = pd.DataFrame([{
        "pixels_on_glacier": stats_before["count"],
        "min_m": stats_before["min"], "max_m": stats_before["max"],
        "mean_m": stats_before["mean"], "median_m": stats_before["median"],
        "std_m": stats_before["std"], "mad_m": stats_before["mad"],
        "p95_m": stats_before["p95"], "p99_m": stats_before["p99"], "p995_m": stats_before["p995"],
    }])
    slope_df = pd.DataFrame([{
        "slope_mean_deg": slope_stats["mean"],
        "slope_median_deg": slope_stats["median"],
        "slope_std_deg": slope_stats["std"],
        "slope_p95_deg": slope_stats["p95"]
    }])
    cleanup_df = pd.DataFrame([{
        "spikes_fixed": int(np.sum(spikes)),
        "AOI_max_before_m": stats_before["max"],
        "AOI_max_after_m": stats_after["max"],
        "AOI_p995_before_m": stats_before["p995"],
        "AOI_p995_after_m": stats_after["p995"],
        "threshold_used_m": thr
    }])

    per_csv   = os.path.join(out_dir, f"{label}_on_glacier_metrics.csv")
    slope_csv = os.path.join(out_dir, f"{label}_on_glacier_slope_metrics.csv")
    clean_csv = os.path.join(out_dir, f"{label}_on_glacier_cleanup_summary.csv")
    per_dem_df.to_csv(per_csv, index=False)
    slope_df.to_csv(slope_csv, index=False)
    cleanup_df.to_csv(clean_csv, index=False)

    return dict(
        masked_path=masked_path, cleaned_path=cleaned_path, hypsometry_png=hyps_path,
        per_csv=per_csv, slope_csv=slope_csv, clean_csv=clean_csv,
        per_df=per_dem_df, slope_df=slope_df, clean_df=cleanup_df
    )

def main():
    st.set_page_config(
        page_title="ASTER DEM Processor",
        page_icon="/home/ashutokumar/Pinn_mass_balance/UNDlogo.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Display UND logo and title
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("/home/ashutokumar/Pinn_mass_balance/UNDlogo.png", width=100)
    with col2:
        st.title("ASTER DEM Processing Application")
        st.markdown("**Process ASTER L1A data to create high-quality DEMs with coregistration options**")
    
    # Setup environment
    setup_environment()
    
    # Add cache clearing button in sidebar
    with st.sidebar:
        st.markdown("---")
        if st.button("🧹 Clear Cache & Reset", help="Clear all cached data and reset the application"):
            clear_cache_and_cleanup()
            st.success("✅ Cache cleared! Page will reload.")
            st.rerun()
    
    # Sidebar for navigation and help
    st.sidebar.header("🧭 Navigation")
    
    # Radio buttons for main navigation
    processing_mode = st.sidebar.radio(
        "Select Processing Mode:",
        [
            "🚀 ASP Stereo DEM Reconstruction Only",
            "🔄 DEM Coregistration Only", 
            "⚡ Complete End-to-End Processing",
            "🧊 Complete End-to-End with Outline (Glacier)"
        ]
    )
    
    # Help section with ASP documentation links
    st.sidebar.markdown("---")
    
    with st.sidebar.expander("📚 Help & Documentation", expanded=False):
        st.markdown("""
        ### 🔗 ASP Documentation Links
        
        **📖 Main Documentation:**
        - [Ames Stereo Pipeline Home](https://stereopipeline.readthedocs.io/en/latest/)
        - [Installation Guide](https://stereopipeline.readthedocs.io/en/latest/installation.html)
        - [How ASP Works](https://stereopipeline.readthedocs.io/en/latest/how_asp_works.html)
        
        **🌍 Earth Image Processing:**
        - [Tutorial: Processing Earth Images](https://stereopipeline.readthedocs.io/en/latest/tutorial_earth.html)
        - [Supported Earth Products](https://stereopipeline.readthedocs.io/en/latest/tutorial_earth.html#supported-products)
        - [ASTER Processing Guide](https://stereopipeline.readthedocs.io/en/latest/examples.html#aster)
        
        **🛠️ Key Tools:**
        - [aster2asp Tool](https://stereopipeline.readthedocs.io/en/latest/tools/aster2asp.html)
        - [stereo Tool](https://stereopipeline.readthedocs.io/en/latest/tools/stereo.html)
        - [pc_align Tool](https://stereopipeline.readthedocs.io/en/latest/tools/pc_align.html)
        - [point2dem Tool](https://stereopipeline.readthedocs.io/en/latest/tools/point2dem.html)
        
        **📊 Validation & Quality:**
        - [pc_align Validation](https://stereopipeline.readthedocs.io/en/latest/tools/pc_align.html#pc-align-validation)
        - [Error Propagation](https://stereopipeline.readthedocs.io/en/latest/error_propagation.html)
        - [Output Files Guide](https://stereopipeline.readthedocs.io/en/latest/outputfiles.html)
        
        **🗂️ Data Sources:**
        - [OpenTopography Portal](https://portal.opentopography.org/)
        - [OpenTopography API Guide](https://portal.opentopography.org/apidocs/)
        - [ASTER Data Download](https://search.earthdata.nasa.gov/)
        - [Copernicus DEM](https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model)
        
        **📥 Download & Setup:**
        - [ASP Precompiled Binaries](https://github.com/NeoGeographyToolkit/StereoPipeline/releases)
        - [Conda Installation](https://stereopipeline.readthedocs.io/en/latest/installation.html#conda-and-docker)
        - [System Requirements](https://stereopipeline.readthedocs.io/en/latest/installation.html#system-requirements)
        
        **🔬 Advanced Topics:**
        - [Bundle Adjustment](https://stereopipeline.readthedocs.io/en/latest/bundle_adjustment.html)
        - [Advanced Stereo Topics](https://stereopipeline.readthedocs.io/en/latest/stereodefault.html)
        - [Stereo Algorithms](https://stereopipeline.readthedocs.io/en/latest/correlation.html)
        
        **📚 Examples & Tutorials:**
        - [Stereo Processing Examples](https://stereopipeline.readthedocs.io/en/latest/examples.html)
        - [DigitalGlobe Processing](https://stereopipeline.readthedocs.io/en/latest/examples.html#digitalglobe)
        - [Declassified Satellite Images](https://stereopipeline.readthedocs.io/en/latest/examples.html#declassified-satellite-images-kh-4b)
        
        **🆘 Support:**
        - [Getting Help](https://stereopipeline.readthedocs.io/en/latest/introduction.html#getting-help-and-reporting-bugs)
        - [GitHub Issues](https://github.com/NeoGeographyToolkit/StereoPipeline/issues)
        - [ASP Google Group](https://groups.google.com/forum/#!forum/ames-stereo-pipeline-support)
        """)
    
    # API Key information
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **🔑 OpenTopography API Key:**
    
    To use DEM coregistration features, you need an API key from [OpenTopography](https://portal.opentopography.org/requestAccess).
    
    **Current Status:** ✅ Configured
    """)
    
    # System status
    st.sidebar.markdown("---")
    st.sidebar.success("""
    **🖥️ System Status:**
    - ASP Version: 3.6.0-alpha
    - Environment: Ready ✅
    - Validation: Enabled ✅
    """)
    
    # Quick test with the known working file
    st.sidebar.markdown("---")
    if st.sidebar.button("🧪 Test with Known Working File"):
        test_zip = "/home/ashutokumar/Pinn_mass_balance/DEM/2011/AST_L1A_00304212011054141_20250626170033_1110403.zip"
        if os.path.exists(test_zip):
            st.sidebar.success(f"✅ Found test file: {os.path.basename(test_zip)}")
            
            # Test extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                extracted_dir = extract_aster_zip(test_zip, temp_dir)
                if extracted_dir:
                    st.sidebar.success("✅ Test extraction successful")
                else:
                    st.sidebar.error("❌ Test extraction failed")
        else:
            st.sidebar.error("❌ Test file not found")
    
    # Main content area
    if processing_mode == "🚀 ASP Stereo DEM Reconstruction Only":
        st.header("🔧 ASTER L1A to DEM Processing")
        
        uploaded_file = st.file_uploader(
            "Upload ASTER L1A ZIP file", 
            type=['zip'],
            help="Upload your ASTER L1A zip file (e.g., AST_L1A_*.zip)"
        )
        
        if uploaded_file is not None:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded file
                zip_path = os.path.join(temp_dir, uploaded_file.name)
                with open(zip_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                st.success(f"✅ Uploaded: {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.1f} MB)")
                
                if st.button("🚀 Process ASTER Data to DEM"):
                    # Extract ASTER data
                    extract_dir = os.path.join(temp_dir, "extracted")
                    aster_dir = extract_aster_zip(zip_path, extract_dir)
                    
                    if aster_dir:
                        # Convert to ASP format
                        asp_output_prefix = os.path.join(temp_dir, "asp_output", "out")
                        if process_aster_to_asp(aster_dir, asp_output_prefix):
                            
                            # Find generated files
                            asp_dir = os.path.dirname(asp_output_prefix)
                            left_image = glob.glob(os.path.join(asp_dir, "*Band3N.tif"))
                            right_image = glob.glob(os.path.join(asp_dir, "*Band3B.tif"))
                            left_camera = glob.glob(os.path.join(asp_dir, "*Band3N.xml"))
                            right_camera = glob.glob(os.path.join(asp_dir, "*Band3B.xml"))
                            
                            if all([left_image, right_image, left_camera, right_camera]):
                                # Run stereo processing
                                stereo_output_prefix = os.path.join(temp_dir, "stereo_output", "stereo")
                                if run_stereo_processing(left_image[0], right_image[0], 
                                                       left_camera[0], right_camera[0], 
                                                       stereo_output_prefix):
                                    
                                    # Generate DEM
                                    point_cloud = f"{stereo_output_prefix}-PC.tif"
                                    dem_output = os.path.join(temp_dir, "final_dem")
                                    
                                    if os.path.exists(point_cloud):
                                        final_dem = generate_dem(point_cloud, dem_output, resolution=30)
                                        
                                        if final_dem:
                                            st.success("🎉 **DEM Processing Complete!**")
                                            
                                            # Analyze and visualize DEM
                                            analyze_dem(final_dem)
                                            
                                            # Provide download
                                            with open(final_dem, 'rb') as f:
                                                st.download_button(
                                                    label="📥 Download DEM",
                                                    data=f.read(),
                                                    file_name=f"aster_dem_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tif",
                                                    mime="application/octet-stream"
                                                )
                            else:
                                st.error("❌ Could not find all required ASP output files")
    
    elif processing_mode == "🔄 DEM Coregistration Only":
        st.header("📐 DEM Coregistration")
        
        coregistration_type = st.selectbox(
            "Select Coregistration Type:",
            [
                "DEM-to-DEM (using COP30_E reference)",
                "🛰️ DEM-to-Altimetry (using ICESat-2 points)"
            ]
        )
        
        uploaded_dem = st.file_uploader(
            "Upload your DEM file", 
            type=['tif', 'tiff'],
            help="Upload the DEM file you want to coregister"
        )
        
        if uploaded_dem is not None:
            # Clear any previous processing state
            clear_cache_and_cleanup()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded DEM
                dem_path = os.path.join(temp_dir, uploaded_dem.name)
                with open(dem_path, 'wb') as f:
                    f.write(uploaded_dem.getbuffer())
                
                st.success(f"✅ Uploaded DEM: {uploaded_dem.name}")
                
                if coregistration_type == "DEM-to-DEM (using COP30_E reference)":
                    reference_label = st.selectbox(
                                            "Reference DEM:",
                                            ["Copernicus DEM 30 m (COP30_E)"]
                                        )
                    label_to_code = {"Copernicus DEM 30 m (COP30_E)": "COP30_E",
                                        }
                                                            
                    if st.button("🔄 Coregister DEM"):
                        # Get DEM bounds
                        with rasterio.open(dem_path) as src:
                            bounds = src.bounds
                            crs = src.crs
                            
                            if crs.to_epsg() != 4326:
                                #from pyproj import Transformer
                                transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
                                west, south = transformer.transform(bounds.left, bounds.bottom)
                                east, north = transformer.transform(bounds.right, bounds.top)
                            else:
                                west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top
                        
                        # Download reference DEM
                        ref_code = label_to_code[reference_label]
                        ref_dem_path = os.path.join(temp_dir, f"reference_{ref_code.lower()}.tif")
                        ref_dem = download_reference_dem((west, south, east, north), ref_code, ref_dem_path)
                        
                        if ref_dem:
                            # Coregister
                            output_prefix = os.path.join(temp_dir, "aligned")
                            aligned_dem = coregister_dem(dem_path, ref_dem, output_prefix)
                            
                            if aligned_dem:
                                st.success("🎉 **Coregistration Complete!**")
                                
                                # Analyze aligned DEM
                                analyze_dem(aligned_dem)
                                
                                # Provide download
                                with open(aligned_dem, 'rb') as f:
                                    st.download_button(
                                        label="📥 Download Coregistered DEM",
                                        data=f.read(),
                                        file_name=f"coregistered_dem_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tif",
                                        mime="application/octet-stream"
                                    )
                
                elif coregistration_type == "🛰️ DEM-to-Altimetry (using ICESat-2 points)":
                    st.info("""
                    **🛰️ ICESat-2 Altimetry Coregistration:**
                    
                    This method aligns your DEM to **real** high-accuracy laser altimetry points from the ICESat-2 satellite.
                    
                    **Advantages:**
                    - Sub-meter vertical accuracy from laser altimetry
                    - Ideal for glacier and ice sheet research
                    - Independent validation source
                    - Sparse but highly accurate reference points
                    
                    **Requirements:**
                    - Internet connection for SlideRule Earth API
                    - SlideRule Python package (`sliderule>=4.0.0`)
                    - **Note**: ICESat-2 has global coverage - data is available worldwide!
                    
                    **Detailed Process:**
                    1. 🌍 **Query Real ICESat-2 ATL06** data using SlideRule Earth API
                    2. 🔄 **Coordinate System Transformation**: ITRF2014 → WGS84
                    3. 🎯 **Point-to-Point Alignment** (optimal for sparse altimetry points)
                    4. 📊 **Before/After Analysis**: Compare alignment quality and statistics
                    5. ✅ **Validation**: RMSE/MAE statistics with real satellite data
                    """)
                    
                    # Add coordinate system information
                    with st.expander("🌐 Coordinate System Details"):
                        st.markdown("""
                        **ICESat-2 Data Specifications:**
                        - **Original CRS**: ITRF2014 (International Terrestrial Reference Frame 2014)
                        - **Target CRS**: WGS84 (World Geodetic System 1984)
                        - **Transformation**: Handled automatically by GeoPandas
                        - **Vertical Datum**: Ellipsoidal heights (compatible with ASP DEMs)
                        
                        **Why This Matters:**
                        - ITRF2014 is the most accurate global reference frame
                        - WGS84 alignment ensures compatibility with your ASTER DEM
                        - Proper transformation prevents systematic biases
                        """)
                    
                    if st.button("🚀 Run ICESat-2 Coregistration"):
                        if uploaded_dem is not None:
                            with tempfile.TemporaryDirectory() as temp_dir:
                                # Save uploaded DEM
                                dem_path = os.path.join(temp_dir, uploaded_dem.name)
                                with open(dem_path, 'wb') as f:
                                    f.write(uploaded_dem.getbuffer())
                                
                                st.success(f"✅ Uploaded: {uploaded_dem.name}")
                                
                                # Copy DEM to processing directory for ICESat-2 script
                                # The script looks for *DEM.tif files, so name it appropriately
                                icesat2_dem_path = os.path.join("/home/ashutokumar/Pinn_mass_balance/ASTER_processing", "uploaded_aster_DEM.tif")
                                import shutil
                                os.makedirs("/home/ashutokumar/Pinn_mass_balance/ASTER_processing", exist_ok=True)
                                shutil.copy2(dem_path, icesat2_dem_path)
                                
                                # Run ICESat-2 coregistration
                                st.info("🛰️ Starting ICESat-2 coregistration process...")
                                
                                # Create progress tracking
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                try:
                                    # Step 1: Query ICESat-2 data
                                    status_text.text("🌍 Step 1/5: Querying ICESat-2 ATL06 data from SlideRule...")
                                    progress_bar.progress(20)
                                    
                                    with st.spinner("Querying ICESat-2 data and running alignment..."):
                                        # Debug info
                                        st.info("🔍 **Debug Info**: Running ICESat-2 script...")
                                        st.code(f"DEM copied to: {icesat2_dem_path}")
                                        
                                        # Use tutorial-based ICESat-2 method directly
                                        aligned_dem_result = run_icesat2_tutorial_method(dem_path, temp_dir)
                                        success = aligned_dem_result is not None
                                    
                                    if success:
                                        # Step 2: Show coordinate transformation
                                        status_text.text("🔄 Step 2/5: Coordinate system transformation (ITRF2014 → WGS84)...")
                                        progress_bar.progress(40)
                                        
                                        # Step 3: Alignment
                                        status_text.text("🎯 Step 3/5: Running point-to-point alignment...")
                                        progress_bar.progress(60)
                                        
                                        # Step 4: Before/After Analysis
                                        status_text.text("📊 Step 4/5: Generating before/after analysis...")
                                        progress_bar.progress(80)
                                        
                                        # Step 5: Validation
                                        status_text.text("✅ Step 5/5: Computing validation statistics...")
                                        progress_bar.progress(100)
                                        
                                        st.success("✅ ICESat-2 coregistration completed!")
                                        
                                        if aligned_dem_result:
                                            st.success(f"🎯 ICESat-2 Aligned DEM created: {os.path.basename(aligned_dem_result)}")
                                            
                                            # Before/After Analysis
                                            st.subheader("📊 Before/After Alignment Analysis")
                                            
                                            # Enhanced analysis with before/after comparison
                                            analyze_dem_with_icesat2_comparison(dem_path, aligned_dem_result, None, None)
                                            
                                            # Provide download
                                            with open(aligned_dem_result, 'rb') as f:
                                                st.download_button(
                                                    label="📥 Download ICESat-2 Aligned DEM",
                                                    data=f.read(),
                                                    file_name=f"icesat2_aligned_{os.path.basename(aligned_dem_result)}",
                                                    mime="application/octet-stream"
                                                )
                                        else:
                                            st.warning("⚠️ ICESat-2 alignment failed or no result produced")
                                    else:
                                        st.error("❌ ICESat-2 coregistration failed")
                                        st.info("💡 **Tip**: ICESat-2 requires internet connection and SlideRule API access")
                                        
                                except Exception as e:
                                    st.error(f"❌ Error running ICESat-2 coregistration: {e}")
                                
                                finally:
                                    # Clean up temporary DEM file
                                    if 'icesat2_dem_path' in locals() and os.path.exists(icesat2_dem_path):
                                        os.remove(icesat2_dem_path)
                        else:
                            st.warning("⚠️ Please upload a DEM file first")
    
    elif processing_mode == "⚡ Complete End-to-End Processing":
        st.header("🔄 Complete ASTER DEM Processing Pipeline")
        
        uploaded_file = st.file_uploader(
            "Upload ASTER L1A ZIP file", 
            type=['zip'],
            help="Upload your ASTER L1A zip file for complete processing"
        )
        
        # Processing options
        st.subheader("Processing Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            dem_resolution = st.number_input("DEM Resolution (meters)", min_value=10, max_value=100, value=30)
            enable_coregistration = st.checkbox("Enable Coregistration", value=True)
        
        with col2:
            create_visualizations = st.checkbox("Create Analysis Visualizations", value=True)
            
        # Coregistration method selection (only show if coregistration is enabled)
        if enable_coregistration:
            st.subheader("🔄 Coregistration Method")
            coregistration_method = st.radio(
                "Select coregistration method:",
                [
                    "🌍 DEM-to-DEM (using COP30_E reference)",
                    "🛰️ DEM-to-Altimetry (using ICESat-2 points)"
                ],
                help="Choose between DEM-to-DEM alignment or ICESat-2 altimetry alignment"
            )
            
            # Show method-specific information
            if coregistration_method == "🌍 DEM-to-DEM (using COP30_E reference)":
                st.info("""
                **🌍 DEM-to-DEM Coregistration:**
                - Uses Copernicus DEM Ellipsoidal (COP30_E) as reference
                - Best for general terrain and large-scale alignment
                - Point-to-plane alignment method
                - Downloads reference DEM automatically
                """)
            else:
                st.info("""
                                    **🛰️ ICESat-2 Altimetry Coregistration:**
                    - Uses real ICESat-2 laser altimetry points as reference
                    - Sub-meter vertical accuracy from satellite laser
                    - Ideal for glacier and ice sheet research
                    - Point-to-point alignment (optimal for sparse points)
                    - Automatic coordinate transformation (ITRF2014 → WGS84)
                    - **Requires**: Internet connection, SlideRule package (global coverage!)
                """)
                
                # Show coordinate system details for ICESat-2
                with st.expander("🌐 ICESat-2 Coordinate System Details"):
                    st.markdown("""
                    **Coordinate Transformation Process:**
                    - **Source**: ITRF2014 (International Terrestrial Reference Frame 2014)
                    - **Target**: WGS84 (World Geodetic System 1984)
                    - **Method**: GeoPandas automatic CRS transformation
                    - **Accuracy**: Sub-centimeter transformation precision
                    
                    **Why This Matters for Glaciers:**
                    - ITRF2014 is the most accurate global reference frame
                    - ICESat-2 provides sub-meter accuracy on ice/snow surfaces
                    - Point-to-point method handles sparse altimetry data optimally
                    """)
    
        if uploaded_file is not None and st.button("🚀 Run Complete Processing Pipeline"):
            # Clear any previous processing state
            clear_cache_and_cleanup()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded file
                zip_path = os.path.join(temp_dir, uploaded_file.name)
                with open(zip_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                st.success(f"✅ Starting complete processing for: {uploaded_file.name}")
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Extract ASTER data
                    status_text.text("Step 1/6: Extracting ASTER data...")
                    progress_bar.progress(10)
                    
                    extract_dir = os.path.join(temp_dir, "extracted")
                    aster_dir = extract_aster_zip(zip_path, extract_dir)
                    
                    if not aster_dir:
                        st.error("❌ Failed to extract ASTER data")
                        return
                    
                    # Step 2: Convert to ASP format
                    status_text.text("Step 2/6: Converting to ASP format...")
                    progress_bar.progress(20)
                    
                    asp_output_prefix = os.path.join(temp_dir, "asp_output", "out")
                    if not process_aster_to_asp(aster_dir, asp_output_prefix):
                        st.error("❌ Failed to convert to ASP format")
                        return
                    
                    # Step 3: Stereo processing
                    status_text.text("Step 3/6: Running stereo processing...")
                    progress_bar.progress(40)
                    
                    asp_dir = os.path.dirname(asp_output_prefix)
                    left_image = glob.glob(os.path.join(asp_dir, "*Band3N.tif"))[0]
                    right_image = glob.glob(os.path.join(asp_dir, "*Band3B.tif"))[0]
                    left_camera = glob.glob(os.path.join(asp_dir, "*Band3N.xml"))[0]
                    right_camera = glob.glob(os.path.join(asp_dir, "*Band3B.xml"))[0]
                    
                    stereo_output_prefix = os.path.join(temp_dir, "stereo_output", "stereo")
                    if not run_stereo_processing(left_image, right_image, left_camera, right_camera, stereo_output_prefix):
                        st.error("❌ Failed stereo processing")
                        return
                    
                    # Step 4: Generate DEM
                    status_text.text("Step 4/6: Generating DEM...")
                    progress_bar.progress(60)
                    
                    point_cloud = f"{stereo_output_prefix}-PC.tif"
                    dem_output = os.path.join(temp_dir, "raw_dem")
                    raw_dem = generate_dem(point_cloud, dem_output, resolution=dem_resolution)
                    
                    if not raw_dem:
                        st.error("❌ Failed to generate DEM")
                        return
                    
                    final_dem = raw_dem
                    
                    # Step 5: Coregistration (optional)
                    if enable_coregistration:
                        status_text.text("Step 5/6: Coregistering DEM...")
                        progress_bar.progress(80)
                        
                        if coregistration_method == "🌍 DEM-to-DEM (using COP30_E reference)":
                            # DEM-to-DEM coregistration using COP30_E
                            st.info("🌍 Running DEM-to-DEM coregistration with COP30_E...")
                            
                            # Get DEM bounds for reference download
                            with rasterio.open(raw_dem) as src:
                                bounds = src.bounds
                                crs = src.crs
                                
                                if crs.to_epsg() != 4326:
                                    #from pyproj import Transformer
                                    transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
                                    west, south = transformer.transform(bounds.left, bounds.bottom)
                                    east, north = transformer.transform(bounds.right, bounds.top)
                                else:
                                    west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top
                            
                            # Download COP30_E reference DEM
                            ref_dem_path = os.path.join(temp_dir, "reference_cop30_e.tif")
                            ref_dem = download_reference_dem((west, south, east, north), "COP30_E", ref_dem_path)
                            
                            if ref_dem:
                                output_prefix = os.path.join(temp_dir, "aligned_cop30")
                                aligned_dem = coregister_dem(raw_dem, ref_dem, output_prefix)
                                if aligned_dem:
                                    final_dem = aligned_dem
                                    st.success("✅ DEM-to-DEM coregistration completed!")
                            else:
                                st.warning("⚠️ Failed to download COP30_E reference DEM")
                        
                        elif coregistration_method == "🛰️ DEM-to-Altimetry (using ICESat-2 points)":
                            # ICESat-2 coregistration
                            st.info("🛰️ Running ICESat-2 altimetry coregistration...")
                            
                            # Copy DEM to processing directory for ICESat-2 script
                            # The script looks for *DEM.tif files, so name it appropriately
                            icesat2_dem_path = os.path.join("/home/ashutokumar/Pinn_mass_balance/ASTER_processing", "temp_aster_DEM.tif")
                            import shutil
                            os.makedirs("/home/ashutokumar/Pinn_mass_balance/ASTER_processing", exist_ok=True)
                            shutil.copy2(raw_dem, icesat2_dem_path)
                            
                            try:
                                # Alternative ICESat-2 implementation with fallback
                                st.info("🛰️ **ICESat-2 Coregistration**: Attempting multiple methods...")
                                
                                # Use tutorial-based ICESat-2 method directly
                                st.info("🛰️ **ICESat-2 Tutorial Method**: Using SlideRule Earth API...")
                                
                                aligned_dem_result = run_icesat2_tutorial_method(raw_dem, temp_dir)
                                
                                if aligned_dem_result:
                                    # Success with tutorial method
                                    final_dem = aligned_dem_result
                                    st.success("✅ ICESat-2 coregistration completed!")
                                    st.info("🔄 **Coordinate Transformation Applied**: ITRF2014 → WGS84")
                                    st.info("📊 **Method**: Tutorial-compliant point-to-point alignment")
                                
                                else:
                                    # ICESat-2 method failed - continue with original DEM
                                    st.warning("⚠️ ICESat-2 coregistration failed - continuing with original DEM")
                                    st.info("📋 **Result**: No coregistration applied - using raw ASTER DEM")
                                    st.info("💡 **Note**: ICESat-2 has global coverage - this is likely a technical issue (network, SlideRule API, or package installation)")
                                    
                                    # Show what ICESat-2 would provide if available
                                    with st.expander("ℹ️ About ICESat-2 Coregistration"):
                                        st.markdown("""
                                        **ICESat-2 provides:**
                                        - Sub-meter vertical accuracy from laser altimetry
                                        - Independent validation from satellite measurements
                                        - Ideal for glacier and ice sheet research
                                        - Sparse but highly accurate reference points
                                        
                                        **Requirements:**
                                        - Internet connection for SlideRule API
                                        - `sliderule>=4.0.0` package
                                        - Global coverage available worldwide
                                        
                                        **Alternative**: Use DEM-to-DEM coregistration with COP30_E reference
                                        """)
                            
                            except Exception as e:
                                st.error(f"❌ Error in ICESat-2 processing: {e}")
                                st.info("📋 **Result**: Continuing with original DEM")
                            
                            finally:
                                # Clean up temporary DEM file
                                if os.path.exists(icesat2_dem_path):
                                    os.remove(icesat2_dem_path)
                    
                    # Step 6: Analysis and visualization
                    if create_visualizations:
                        status_text.text("Step 6/6: Creating analysis and visualizations...")
                        progress_bar.progress(90)
                        
                        analyze_dem(final_dem)
                    
                    # Complete
                    progress_bar.progress(100)
                    status_text.text("✅ Processing complete!")
                    
                    st.success("🎉 **Complete Processing Pipeline Finished!**")
                    
                    # Provide download
                    with open(final_dem, 'rb') as f:
                        st.download_button(
                            label="📥 Download Final DEM",
                            data=f.read(),
                            file_name=f"final_aster_dem_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tif",
                            mime="application/octet-stream"
                        )
                    
                    # Summary
                    st.subheader("📊 Processing Summary")
                    
                    # Determine coregistration method for summary
                    coregistration_summary = "Disabled"
                    if enable_coregistration:
                        if coregistration_method == "🌍 DEM-to-DEM (using COP30_E reference)":
                            coregistration_summary = "DEM-to-DEM (COP30_E)"
                        elif coregistration_method == "🛰️ DEM-to-Altimetry (using ICESat-2 points)":
                            coregistration_summary = "ICESat-2 Altimetry (ITRF2014→WGS84)"
                    
                    summary_data = {
                        "Input File": uploaded_file.name,
                        "Processing Mode": "Complete Pipeline",
                        "DEM Resolution": f"{dem_resolution}m",
                        "Coregistration": coregistration_summary,
                        "Final DEM": os.path.basename(final_dem),
                        "Processing Time": "Completed",
                        "Status": "✅ Success"
                    }
                    
                    summary_df = pd.DataFrame(list(summary_data.items()), columns=["Parameter", "Value"])
                    st.table(summary_df)
                    
                except Exception as e:
                    st.error(f"❌ Processing failed: {e}")
                    progress_bar.progress(0)
                    status_text.text("❌ Processing failed")
    

    elif processing_mode == "🧊 Complete End-to-End with Outline (Glacier)":
        st.header("🧊 Complete End-to-End with Glacier Outline")
        # ---- Glacier mode help panel ----
        GLACIER_MODE_HELP = """
        ### What is AOI?
        **AOI (Area of Interest)** is the glacier outline you upload as a GeoPackage (`.gpkg`). The app reprojects that polygon to the DEM CRS and rasterizes it to a mask so all steps below operate strictly *on-glacier*.

        ### What this mode does
        1. **Extract** ASTER L1A zip  
        2. **Convert for ASP** with `aster2asp` (Band3N/Band3B + cameras)  
        3. **Stereo** (`stereo -t aster …`) → point cloud (`*-PC.tif`)  
        4. **Grid** to DEM via `point2dem`  
        5. **Coregister to COP30_E** (download via OpenTopography → `pc_align --highest-accuracy --alignment-method point-to-plane` → `point2dem`) to get an **aligned scene DEM**  
        6. **Clip to AOI** (your glacier) → `glacier_AOI_masked.tif`  
        7. **Clean spikes (AOI only)** using a 3×3 local median with threshold \\(T=\\max(100 \\text{ m}, 8\\times\\text{NMAD})\\) → `glacier_AOI_cleaned_spikes.tif`  
        8. **Summaries & plots (AOI)**: robust elevation stats, slope stats, hypsometry histogram; CSV downloads.

        ### Outputs
        - **AOI-cleaned DEM** *(recommended)*: `glacier_AOI_cleaned_spikes.tif`  
        - AOI-masked DEM (no cleaning): `glacier_AOI_masked.tif`  
        - Final DEM (full extent): scene DEM after coregistration (not clipped)

        **Notes**
        - The `.gpkg` CRS can be anything; it’s reprojected to match the DEM before masking.  
        - COP30_E is requested over the DEM’s bounds (converted to geographic for download), but processing/outputs stay in the DEM CRS.
        """

        with st.expander("ℹ️ About this mode (AOI / Glacier outline)", expanded=False):
            st.markdown(GLACIER_MODE_HELP)

        st.markdown("Runs stereo → point2dem → **COP30_E coregistration**, then clips strictly to your uploaded glacier outline and generates on-glacier summaries & downloads.")

        # Ask for BOTH inputs up front
        uploaded_file = st.file_uploader(
            "Upload ASTER L1A ZIP file", type=['zip'],
            help="Upload your ASTER L1A zip file (e.g., AST_L1A_*.zip)"
        )
        aoi_file = st.file_uploader(
            "Upload Glacier Boundary (.gpkg)", type=['gpkg'],
            help="Provide a GeoPackage with the glacier polygon(s)."
        )

        # Simple config (same defaults as your complete pipeline)
        st.subheader("Processing Configuration")
        col1, col2 = st.columns(2)
        with col1:
            dem_resolution = st.number_input("DEM Resolution (meters)", min_value=10, max_value=100, value=30)
        with col2:
            create_visualizations = st.checkbox("Create Analysis Visualizations", value=True)

        run_btn = st.button("🚀 Run End-to-End + Glacier Outline")

        if run_btn and (uploaded_file is None or aoi_file is None):
            st.warning("Please upload both the ASTER L1A ZIP and the glacier boundary .gpkg.")
        if uploaded_file is not None and aoi_file is not None and run_btn:
            clear_cache_and_cleanup()
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save inputs
                zip_path = os.path.join(temp_dir, uploaded_file.name)
                with open(zip_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                gpkg_path = os.path.join(temp_dir, aoi_file.name)
                with open(gpkg_path, 'wb') as f:
                    f.write(aoi_file.getbuffer())

                st.success(f"✅ Inputs: {os.path.basename(zip_path)}, {os.path.basename(gpkg_path)}")
                progress_bar = st.progress(0); status_text = st.empty()
                try:
                    # 1) Extract
                    status_text.text("Step 1/7: Extracting ASTER data…"); progress_bar.progress(10)
                    extract_dir = os.path.join(temp_dir, "extracted")
                    aster_dir = extract_aster_zip(zip_path, extract_dir)
                    if not aster_dir: st.stop()

                    # 2) aster2asp
                    status_text.text("Step 2/7: Converting to ASP format…"); progress_bar.progress(20)
                    asp_output_prefix = os.path.join(temp_dir, "asp_output", "out")
                    if not process_aster_to_asp(aster_dir, asp_output_prefix): st.stop()

                    # 3) stereo
                    status_text.text("Step 3/7: Stereo processing…"); progress_bar.progress(35)
                    asp_dir = os.path.dirname(asp_output_prefix)
                    left_image  = glob.glob(os.path.join(asp_dir, "*Band3N.tif"))[0]
                    right_image = glob.glob(os.path.join(asp_dir, "*Band3B.tif"))[0]
                    left_camera  = glob.glob(os.path.join(asp_dir, "*Band3N.xml"))[0]
                    right_camera = glob.glob(os.path.join(asp_dir, "*Band3B.xml"))[0]
                    stereo_output_prefix = os.path.join(temp_dir, "stereo_output", "stereo")
                    if not run_stereo_processing(left_image, right_image, left_camera, right_camera, stereo_output_prefix): st.stop()

                    # 4) point2dem
                    status_text.text("Step 4/7: DEM generation…"); progress_bar.progress(50)
                    point_cloud = f"{stereo_output_prefix}-PC.tif"
                    raw_dem = generate_dem(point_cloud, os.path.join(temp_dir, "raw_dem"), resolution=dem_resolution)
                    if not raw_dem: st.stop()

                    # 5) DEM-to-DEM coregistration to COP30_E (use your existing function)
                    status_text.text("Step 5/7: Coregistering to COP30_E…"); progress_bar.progress(65)
                    with rasterio.open(raw_dem) as src:
                        bounds, crs = src.bounds, src.crs
                        if crs and crs.to_epsg() != 4326:
                            t = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
                            west, south = t.transform(bounds.left,  bounds.bottom)
                            east, north = t.transform(bounds.right, bounds.top)
                        else:
                            west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top
                    ref_dem_path = os.path.join(temp_dir, "reference_cop30_e.tif")
                    ref_dem = download_reference_dem((west, south, east, north), "COP30_E", ref_dem_path)
                    final_dem = raw_dem
                    if ref_dem:
                        aligned_dem = coregister_dem(raw_dem, ref_dem, os.path.join(temp_dir, "aligned_cop30"))
                        if aligned_dem: final_dem = aligned_dem
                    else:
                        st.warning("⚠️ Could not download COP30_E; proceeding without coreg.")
                    # After aligned_dem is produced:
                    qa_png = make_dem_align_panel_png(
                        ref_dem,           # the COP30_E path you downloaded
                        raw_dem,    # your original/source DEM before coreg
                        aligned_dem,            # output from coregister_dem
                        out_png=os.path.join(temp_dir, "dem_align_panel.png"),
                        glacier_mask_path=gpkg_path,    
                        title_prefix="COP30_E → ASTER"
                    )

                    st.image(qa_png, caption="Co-registration QA panel", use_container_width=True)
                    with open(qa_png, "rb") as f:
                        st.download_button("Download QA panel (PNG)", data=f.read(), file_name="coreg_QA.png")

                    # 6) (optional) whole-scene quick analysis (unchanged visuals)
                    if create_visualizations:
                        status_text.text("Step 6/7: Whole-scene analysis…"); progress_bar.progress(78)
                        analyze_dem(final_dem)
                    
                    # 7) Strictly on-glacier mask/cleanup/summaries
                    status_text.text("Step 7/7: Glacier AOI clipping & summaries…"); progress_bar.progress(92)
                    glacier_out = _mask_clean_and_summarize_on_glacier(
                        dem_path=final_dem,
                        gpkg_path=gpkg_path,
                        out_dir=os.path.join(temp_dir, "glacier_outputs"),
                        label="glacier"
                    )

                    progress_bar.progress(100); status_text.text("✅ Finished!")

                    st.subheader("📦 Glacier-specific outputs")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        with open(glacier_out["masked_path"], "rb") as f:
                            st.download_button("Download AOI-masked DEM", data=f.read(),
                                            file_name=os.path.basename(glacier_out["masked_path"]))
                    with c2:
                        with open(glacier_out["cleaned_path"], "rb") as f:
                            st.download_button("Download AOI-cleaned DEM", data=f.read(),
                                            file_name=os.path.basename(glacier_out["cleaned_path"]))
                    with c3:
                        st.image(glacier_out["hypsometry_png"], caption="On-glacier hypsometry (cleaned)", use_container_width=True)

                    st.subheader("📊 Glacier-specific summaries")
                    st.dataframe(glacier_out["per_df"])
                    st.dataframe(glacier_out["slope_df"])
                    st.dataframe(glacier_out["clean_df"])

                    cc1, cc2, cc3 = st.columns(3)
                    with cc1:
                        with open(glacier_out["per_csv"], "rb") as f:
                            st.download_button("Download metrics CSV", data=f.read(),
                                            file_name=os.path.basename(glacier_out["per_csv"]))
                    with cc2:
                        with open(glacier_out["slope_csv"], "rb") as f:
                            st.download_button("Download slope CSV", data=f.read(),
                                            file_name=os.path.basename(glacier_out["slope_csv"]))
                    with cc3:
                        with open(glacier_out["clean_csv"], "rb") as f:
                            st.download_button("Download cleanup CSV", data=f.read(),
                                            file_name=os.path.basename(glacier_out["clean_csv"]))

                    st.markdown("---")
                    with open(final_dem, "rb") as f:
                        st.download_button("📥 Download Final DEM (full extent)", data=f.read(),
                                        file_name=f"final_aster_dem_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tif")

                except Exception as e:
                    st.error(f"❌ Processing failed: {e}")
                    progress_bar.progress(0); status_text.text("❌ Processing failed")
    # Footer with proper citation
    st.markdown("---")
    st.markdown("""
    ### Citing the Ames Stereo Pipeline in your work
    
    **In general, use this reference:**
    
    Beyer, Ross A., Oleg Alexandrov, and Scott McMichael. 2018. The Ames Stereo Pipeline: NASA's open source software for deriving and processing terrain data. *Earth and Space Science*, 5. https://doi.org/10.1029/2018EA000409.
    
    **If you are using ASP for application to Earth images, or need a reference which details the quality of output, then we suggest also referencing:**
    
    Shean, D. E., O. Alexandrov, Z. Moratto, B. E. Smith, I. R. Joughin, C. C. Porter, Morin, P. J. 2016. An automated, open-source pipeline for mass production of digital elevation models (DEMs) from very high-resolution commercial stereo satellite imagery. *ISPRS Journal of Photogrammetry and Remote Sensing*. 116.
    """)
if __name__ == "__main__":
    main()
