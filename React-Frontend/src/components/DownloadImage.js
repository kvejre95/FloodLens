import React, { useState } from 'react';
import { TextField, Button, Box, Typography, Container, Paper, Grid } from '@mui/material';
import load_svg from '../infinite-spinner.svg';

function DownloadImage() {
    const [location, setLocation] = useState('');
    const [latitude, setLatitude] = useState('');
    const [longitude, setLongitude] = useState('');
    const [imageUrl, setImageUrl] = useState('');
    const [loading, setLoading] = useState(false);

    const geocode_location = async () => {
        const headers = {
          'User-Agent': 'FloodLense'
        };
        const base_url = 'https://nominatim.openstreetmap.org/search';
        const params = new URLSearchParams({ q: location, format: 'json' });
      
        try {
          const response = await fetch(`${base_url}?${params}`, { headers });
          const data = await response.json();
          if (data && data.length > 0) {
            const { lat, lon } = data[0];
            setLatitude(lat);
            setLongitude(lon);
          } else {
            alert('Location not found');
          }
        } catch (error) {
          console.error('Error:', error);
          alert('Failed to get location info');
        }
      };
      

    const handleDownloadImage = async () => {
        setLoading(true);
        try {
            // First API call to get the image ID
            const response = await fetch(`http://localhost:5000/download_image?latitude=${latitude}&longitude=${longitude}`);
            const data = await response.json();
            console.log(data)
            const image_url = data.url; // Assuming the response contains the URL
            
            // Second API call to get the actual image
            const imageResponse = await fetch(image_url);
            const imageBlob = await imageResponse.blob();
            const imageObjectURL = URL.createObjectURL(imageBlob);
            setImageUrl(imageObjectURL);
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to fetch image');
        }finally{
            setLoading(false);
        }
    };

    return (
        <Container maxWidth="sm">
            <Paper elevation={3} sx={{ p: 4, mt: 4 }}>
                <Typography variant="h5" gutterBottom>
                    Flood Detector
                </Typography>
                <Box component="form" noValidate autoComplete="off" sx={{ mb: 2 }}>
                    <Grid container spacing={2}>
                        <Grid item xs={14} sm={7}>
                            <TextField
                                fullWidth
                                label="Location"
                                value={location}
                                onChange={(e) => setLocation(e.target.value)}
                                variant="outlined"
                                placeholder="Enter Location"
                            />
                        </Grid>
                        <Grid item xs={10} sm={5}>
                            <Button
                                fullWidth
                                variant="contained"
                                color="primary"
                                sx ={{height:55}}
                                onClick={geocode_location}
                            >
                                Get Co-Ordinates
                            </Button>
                        </Grid>
                    </Grid>
                </Box>
                <Box component="form" noValidate autoComplete="off" sx={{ mb: 2 }}>
                    <Grid container spacing={2}>
                        <Grid item xs={12} sm={6}>
                            <TextField
                                fullWidth
                                label="Latitude"
                                value={latitude}
                                onChange={(e) => setLatitude(e.target.value)}
                                variant="outlined"
                                placeholder="Enter latitude"
                            />
                        </Grid>
                        <Grid item xs={12} sm={6}>
                            <TextField
                                fullWidth
                                label="Longitude"
                                value={longitude}
                                onChange={(e) => setLongitude(e.target.value)}
                                variant="outlined"
                                placeholder="Enter longitude"
                            />
                        </Grid>
                    </Grid>
                </Box>
                <Button
                    fullWidth
                    variant="contained"
                    color="primary"
                    sx ={{height:55}}
                    onClick={handleDownloadImage}
                >
                    Display Image
                </Button>
                {loading ? (
                    <Box mt={2} display="flex" justifyContent="center">
                        <img src={load_svg} alt="Loading" />
                    </Box>
                ) : imageUrl && (
                    <Box mt={2} display="flex" justifyContent="center">
                        <img src={imageUrl} alt="Map" style={{ maxWidth: '100%', height: 'auto' }} />
                    </Box>
                )}
            </Paper>
        </Container>
    );
}

export default DownloadImage;
