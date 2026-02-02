# Professional DEM Reconstruction & Coregistration

🏔️ A comprehensive Streamlit application for worldwide Digital Elevation Model (DEM) processing using NASA's Ames Stereo Pipeline (ASP).

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## ✨ Features

### 🛰️ NASA Ames Stereo Pipeline (ASP) DEM Reconstruction
- Process ASTER L1A satellite imagery into high-quality DEMs
- Supports single or multiple file processing with automatic merging
- Configurable output resolution and coordinate systems
- Real-time progress monitoring

### 📐 Advanced DEM Coregistration
- **COP30**: Global coverage using Copernicus DEM (recommended)
- **ICESat-2**: High-accuracy laser altimetry where available
- **Ensemble**: Combined COP30 + ICESat-2 for maximum accuracy
- Adaptive pre-filtering and outlier detection

### 🌍 Professional Tools for Glaciologists
- Worldwide compatibility for any geographic region
- Validation reports with RMSE, MAE, and bias statistics
- Interactive data coverage maps
- Downloadable results in standard GeoTIFF format

## 🔧 Technology Stack

- **Frontend**: Streamlit with interactive maps (Folium)
- **Processing**: NASA Ames Stereo Pipeline (ASP)
- **Geospatial**: GDAL, Rasterio, GeoPandas
- **APIs**: OpenTopography, SlideRule Earth (ICESat-2)
- **Visualization**: Plotly, Matplotlib

## 📦 Installation & Local Development

### Prerequisites
- Python 3.9+
- GDAL system libraries
- NASA ASP binaries (auto-installed in cloud)

### Local Setup
```bash
# Clone repository
git clone https://github.com/nvashutoshkumar/professional-dem-app.git
cd professional-dem-app

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run professional_dem_app.py
```

## ☁️ Cloud Deployment

### Streamlit Cloud
1. Fork this repository
2. Connect to [Streamlit Cloud](https://share.streamlit.io/)
3. Set environment variable: `OPENTOPOGRAPHY_API_KEY`
4. Deploy!

### Environment Variables
```
OPENTOPOGRAPHY_API_KEY=your_api_key_here
```

## 📚 Documentation & Tutorials

- [NASA ASP Documentation](https://stereopipeline.readthedocs.io/)
- [ASP Tutorial Notebooks](https://ideal-xylophone-7vx6j5rvj9jfr64j.github.dev/)
- [UW-Cryo ASP Tutorials](https://github.com/uw-cryo/asp_tutorials)
- [OpenTopography Portal](https://portal.opentopography.org/)

## 🎯 Use Cases

### High Mountain Asia Glacier Research
- Process ASTER stereo imagery for glacier elevation change studies
- Coregister DEMs to global reference datasets
- Generate accuracy reports for scientific publications

### Global DEM Processing
- Flexible coordinate system support (UTM, WGS84)
- Batch processing capabilities for large datasets
- Professional-grade validation and quality control

## 🔬 Scientific Background

This application implements workflows from peer-reviewed NASA ASP tutorials:
- ASTER stereo reconstruction following NASA best practices
- DEM-to-DEM coregistration using Copernicus reference
- DEM-to-altimetry alignment with ICESat-2 laser data
- Comprehensive validation following ASP documentation

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines and submit pull requests.

## 📧 Contact

**Ashutosh Kumar** - ashutosh.kumar@und.edu

---

*Built with ❤️ for the global glaciology and remote sensing community*
