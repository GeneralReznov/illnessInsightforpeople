let map;
let userLocation = null;
let markers = [];
let userMarker = null;

// Initialize the map with Leaflet and Geoapify
function initMap() {
    console.log('Medical Facility Finder loaded');
    
    // Initialize Leaflet map
    map = L.map('map').setView([40.7128, -74.0060], 13);

    // Add OpenStreetMap tile layer (free, no API key required)
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 18,
    }).addTo(map);
    
    // Event listeners
    document.getElementById('find-location-btn').addEventListener('click', findUserLocation);
    document.getElementById('search-location-btn').addEventListener('click', searchManualLocation);
    document.getElementById('search-facilities-btn').addEventListener('click', searchNearbyFacilities);
    
    // Enable search on Enter key for manual location input
    document.getElementById('manual-location-input').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            searchManualLocation();
        }
    });
}

// Find user's current location
function findUserLocation() {
    const locationStatus = document.getElementById('location-status');
    const locationText = document.getElementById('location-text');
    const findLocationBtn = document.getElementById('find-location-btn');
    
    if (navigator.geolocation) {
        findLocationBtn.disabled = true;
        findLocationBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Getting Location...';
        
        locationStatus.className = 'alert alert-info';
        locationStatus.classList.remove('d-none');
        locationText.textContent = 'Getting your location...';
        
        navigator.geolocation.getCurrentPosition(
            (position) => {
                userLocation = {
                    lat: position.coords.latitude,
                    lng: position.coords.longitude
                };
                
                // Update map center
                map.setView([userLocation.lat, userLocation.lng], 15);
                
                // Clear existing markers
                clearMarkers();
                
                // Add user location marker
                userMarker = L.marker([userLocation.lat, userLocation.lng], {
                    icon: L.divIcon({
                        html: '<div style="background-color: #007bff; width: 20px; height: 20px; border-radius: 50%; border: 3px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></div>',
                        iconSize: [20, 20],
                        iconAnchor: [10, 10],
                        className: 'user-location-marker'
                    })
                }).addTo(map);
                
                userMarker.bindPopup("Your Location");
                
                locationStatus.className = 'alert alert-success';
                locationText.textContent = `Location found: ${position.coords.latitude.toFixed(6)}, ${position.coords.longitude.toFixed(6)}`;
                
                // Enable search button
                document.getElementById('search-facilities-btn').disabled = false;
                
                findLocationBtn.disabled = false;
                findLocationBtn.innerHTML = '<i class="fas fa-crosshairs me-2"></i>Find My Location';
            },
            (error) => {
                console.error('Geolocation error:', error);
                locationStatus.className = 'alert alert-danger';
                locationStatus.classList.remove('d-none');
                
                switch(error.code) {
                    case error.PERMISSION_DENIED:
                        locationText.textContent = 'Location access denied. Please enable location services.';
                        break;
                    case error.POSITION_UNAVAILABLE:
                        locationText.textContent = 'Location information unavailable.';
                        break;
                    case error.TIMEOUT:
                        locationText.textContent = 'Location request timed out.';
                        break;
                    default:
                        locationText.textContent = 'An unknown error occurred while getting location.';
                        break;
                }
                
                findLocationBtn.disabled = false;
                findLocationBtn.innerHTML = '<i class="fas fa-crosshairs me-2"></i>Find My Location';
            },
            {
                enableHighAccuracy: true,
                timeout: 10000,
                maximumAge: 0
            }
        );
    } else {
        locationStatus.className = 'alert alert-danger';
        locationStatus.classList.remove('d-none');
        locationText.textContent = 'Geolocation is not supported by this browser.';
    }
}

// Search for location manually using address/city/zip
function searchManualLocation() {
    const locationInput = document.getElementById('manual-location-input');
    const searchQuery = locationInput.value.trim();
    
    if (!searchQuery) {
        alert('Please enter an address, city, or ZIP code.');
        return;
    }
    
    let locationStatus = document.getElementById('location-status');
    let locationText = document.getElementById('location-text');
    const searchBtn = document.getElementById('search-location-btn');
    
    // Create status elements if they don't exist
    if (!locationStatus) {
        locationStatus = document.createElement('div');
        locationStatus.id = 'location-status';
        locationStatus.className = 'alert alert-info';
        locationInput.parentNode.parentNode.appendChild(locationStatus);
        
        locationText = document.createElement('small');
        locationText.id = 'location-text';
        locationStatus.appendChild(locationText);
    }
    
    // Show loading state
    searchBtn.disabled = true;
    searchBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
    
    locationStatus.className = 'alert alert-info';
    locationStatus.classList.remove('d-none');
    locationText.textContent = 'Searching for location...';
    
    // First, try to search for hospitals with this name
    const hospitalSearchUrl = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(searchQuery + ' hospital')}&limit=5&addressdetails=1`;
    
    fetch(hospitalSearchUrl)
        .then(response => response.json())
        .then(hospitalData => {
            searchBtn.disabled = false;
            searchBtn.innerHTML = '<i class="fas fa-search"></i>';
            
            if (hospitalData && hospitalData.length > 0) {
                // Found hospital(s) - use the first one
                const hospital = hospitalData[0];
                userLocation = {
                    lat: parseFloat(hospital.lat),
                    lng: parseFloat(hospital.lon)
                };
                
                // Update map center
                map.setView([userLocation.lat, userLocation.lng], 16);
                
                // Clear existing markers
                clearMarkers();
                
                // Add hospital location marker
                userMarker = L.marker([userLocation.lat, userLocation.lng], {
                    icon: L.divIcon({
                        html: '<div style="background-color: #28a745; width: 20px; height: 20px; border-radius: 50%; border: 3px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></div>',
                        iconSize: [20, 20],
                        iconAnchor: [10, 10],
                        className: 'user-location-marker'
                    })
                }).addTo(map);
                
                // Enable search button
                document.getElementById('search-facilities-btn').disabled = false;
                
                locationStatus.className = 'alert alert-success';
                locationStatus.classList.remove('d-none');
                locationText.textContent = `Hospital found: ${hospital.display_name}`;
                
                // Auto-search for nearby facilities
                searchNearbyFacilities();
                
            } else {
                // No hospital found, try general location search
                const geocodeUrl = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(searchQuery)}&limit=1&addressdetails=1`;
                
                fetch(geocodeUrl)
                    .then(response => response.json())
                    .then(data => {
                        if (data && data.length > 0) {
                            const result = data[0];
                            userLocation = {
                                lat: parseFloat(result.lat),
                                lng: parseFloat(result.lon)
                            };
                            
                            // Update map center
                            map.setView([userLocation.lat, userLocation.lng], 15);
                            
                            // Clear existing markers
                            clearMarkers();
                            
                            // Add user location marker
                            userMarker = L.marker([userLocation.lat, userLocation.lng], {
                                icon: L.divIcon({
                                    html: '<div style="background-color: #007bff; width: 20px; height: 20px; border-radius: 50%; border: 3px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></div>',
                                    iconSize: [20, 20],
                                    iconAnchor: [10, 10],
                                    className: 'user-location-marker'
                                })
                            }).addTo(map);
                            
                            // Enable search button
                            document.getElementById('search-facilities-btn').disabled = false;
                            
                            locationStatus.className = 'alert alert-success';
                            locationStatus.classList.remove('d-none');
                            locationText.textContent = `Location found: ${result.display_name}`;
                            
                        } else {
                            locationStatus.className = 'alert alert-warning';
                            locationStatus.classList.remove('d-none');
                            locationText.textContent = 'Location not found. Please try a different address or hospital name.';
                        }
                    })
                    .catch(error => {
                        console.error('Geocoding error:', error);
                        locationStatus.className = 'alert alert-danger';
                        locationStatus.classList.remove('d-none');
                        locationText.textContent = 'Error searching for location. Please try again.';
                    });
            }
        })
        .catch(error => {
            console.error('Hospital search error:', error);
            searchBtn.disabled = false;
            searchBtn.innerHTML = '<i class="fas fa-search"></i>';
            
            locationStatus.className = 'alert alert-danger';
            locationStatus.classList.remove('d-none');
            locationText.textContent = 'Error searching for location. Please try again.';
        });
}

// Search for nearby medical facilities
function searchNearbyFacilities() {
    if (!userLocation) {
        alert('Please find your location first.');
        return;
    }
    
    const radius = document.getElementById('radius-select').value;
    const facilityType = document.getElementById('facility-type-select').value;
    const loadingSpinner = document.querySelector('.loading-spinner');
    const searchBtn = document.getElementById('search-facilities-btn');
    
    // Show loading state
    loadingSpinner.style.display = 'block';
    searchBtn.disabled = true;
    searchBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Searching...';
    
    // Clear existing facility markers
    clearMarkers();
    
    // Add user location marker back
    if (userMarker) {
        userMarker.addTo(map);
    }
    
    // Fetch nearby facilities with type filter
    fetch(`/api/nearby-facilities?lat=${userLocation.lat}&lng=${userLocation.lng}&radius=${radius}&type=${facilityType}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            loadingSpinner.style.display = 'none';
            searchBtn.disabled = false;
            searchBtn.innerHTML = '<i class="fas fa-search me-2"></i>Search Facilities';
            
            if (data.error) {
                console.error('API Error:', data.error);
                // Show error in status instead of alert
                const locationStatus = document.getElementById('location-status');
                const locationText = document.getElementById('location-text');
                if (locationStatus && locationText) {
                    locationStatus.className = 'alert alert-warning';
                    locationStatus.classList.remove('d-none');
                    locationText.textContent = 'Some facilities may not be available. Showing available results.';
                }
            }
            
            const facilities = data.facilities || [];
            displayFacilities(facilities);
            updateFacilityCount(facilities.length);
        })
        .catch(error => {
            console.error('Error fetching facilities:', error);
            loadingSpinner.style.display = 'none';
            searchBtn.disabled = false;
            searchBtn.innerHTML = '<i class="fas fa-search me-2"></i>Search Facilities';
            
            // Show error in status instead of alert
            const locationStatus = document.getElementById('location-status');
            const locationText = document.getElementById('location-text');
            if (locationStatus && locationText) {
                locationStatus.className = 'alert alert-warning';
                locationStatus.classList.remove('d-none');
                locationText.textContent = 'Unable to load facilities. Please try again or check your connection.';
            }
        });
}

// Global variables for pagination
let allFacilitiesData = [];
let currentPage = 1;
const facilitiesPerPage = 10;

// Display facilities on map and in list with pagination
function displayFacilities(facilities) {
    const container = document.getElementById('facilities-container');
    
    if (facilities.length === 0) {
        container.innerHTML = `
            <div class="col-12 text-center text-muted py-5">
                <i class="fas fa-exclamation-triangle fa-3x mb-3"></i>
                <p>No medical facilities found in the selected radius. Try increasing the search radius.</p>
            </div>
        `;
        document.getElementById('pagination-section').style.display = 'none';
        return;
    }
    
    // Store all facilities data globally
    allFacilitiesData = facilities;
    currentPage = 1;
    
    // Display first page
    displayFacilitiesPage(currentPage);
    
    // Show pagination if more than 15 facilities
    if (facilities.length > facilitiesPerPage) {
        setupPagination();
        document.getElementById('pagination-section').style.display = 'block';
    } else {
        document.getElementById('pagination-section').style.display = 'none';
    }
}

// Display specific page of facilities
function displayFacilitiesPage(page) {
    const container = document.getElementById('facilities-container');
    container.innerHTML = '';
    
    // Clear existing markers
    clearMarkers();
    
    // Add user location marker back
    if (userMarker) {
        userMarker.addTo(map);
    }
    
    const startIndex = (page - 1) * facilitiesPerPage;
    const endIndex = Math.min(startIndex + facilitiesPerPage, allFacilitiesData.length);
    const pageData = allFacilitiesData.slice(startIndex, endIndex);
    
    pageData.forEach((facility, index) => {
        const globalIndex = startIndex + index;
        // Add marker to map
        const lat = facility.geometry.location.lat;
        const lng = facility.geometry.location.lng;
        
        const marker = L.marker([lat, lng], {
            icon: L.divIcon({
                html: `<div style="background-color: #dc3545; width: 24px; height: 24px; border-radius: 50%; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3); display: flex; align-items: center; justify-content: center; color: white; font-size: 12px; font-weight: bold;">${globalIndex + 1}</div>`,
                iconSize: [24, 24],
                iconAnchor: [12, 12],
                className: 'facility-marker'
            })
        }).addTo(map);
        
        markers.push(marker);
        
        // Add click listener for marker
        marker.on('click', () => {
            showFacilityDetails(facility.place_id, facility.name);
        });
        
        // Create facility card
        const categoryName = getCategoryDisplayName(facility.category);
        
        const facilityCard = document.createElement('div');
        facilityCard.className = 'col-12 mb-2';
        // Safely get values to prevent undefined errors
        const facilityName = (facility.name || 'Unnamed Facility').replace(/'/g, "\\'");
        const facilityCategory = facility.category || categoryName || 'Healthcare Facility';
        const facilityAddress = facility.vicinity || facility.address_line1 || 'Address not available';
        const facilityPhone = facility.contact && facility.contact.phone ? facility.contact.phone : '';
        const facilityDistance = facility.distance ? facility.distance : null;
        const facilityHours = facility.opening_hours || '';
        
        facilityCard.innerHTML = `
            <div class="card facility-card h-100 border-0 shadow-sm" style="cursor: pointer; transition: all 0.2s;" 
                 onclick="showFacilityDetails('${facility.place_id || ''}', '${facilityName}')"
                 onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.15)';"
                 onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 2px 8px rgba(0,0,0,0.1)';">
                <div class="card-body p-3">
                    <div class="d-flex align-items-start">
                        <div class="facility-number me-3">
                            <span class="badge bg-danger rounded-pill px-2 py-1">${globalIndex + 1}</span>
                        </div>
                        <div class="flex-grow-1">
                            <h6 class="card-title mb-1 fw-bold text-dark">
                                ${facilityName.replace(/\\'/g, "'")}
                            </h6>
                            <p class="card-text mb-1 text-primary small fw-medium">
                                <i class="fas fa-hospital me-1"></i>${facilityCategory}
                            </p>
                            <p class="card-text mb-2 text-muted small">
                                <i class="fas fa-map-marker-alt me-1"></i>
                                ${facilityAddress}
                            </p>
                            
                            <!-- Additional Details Row -->
                            <div class="d-flex justify-content-between align-items-center">
                                <div class="d-flex flex-column">
                                    ${facilityDistance ? `<small class="text-muted mb-1"><i class="fas fa-route me-1"></i>${facilityDistance} km away</small>` : ''}
                                    ${facilityPhone ? `
                                        <small class="text-success">
                                            <i class="fas fa-phone me-1"></i>${facilityPhone}
                                        </small>
                                    ` : ''}
                                    ${facilityHours ? `
                                        <small class="text-info">
                                            <i class="fas fa-clock me-1"></i>Hours available
                                        </small>
                                    ` : ''}
                                </div>
                                
                                <!-- Action Buttons -->
                                <div class="btn-group btn-group-sm" role="group">
                                    ${facilityPhone ? 
                                      `<a href="tel:${facilityPhone}" class="btn btn-outline-success" title="Call" onclick="event.stopPropagation();">
                                         <i class="fas fa-phone"></i>
                                       </a>` : ''}
                                    <button class="btn btn-outline-primary" 
                                            onclick="event.stopPropagation(); calculateRoute('${facility.place_id || ''}', ${lat}, ${lng})" 
                                            title="Get Directions">
                                        <i class="fas fa-directions"></i>
                                    </button>
                                    <button class="btn btn-outline-info" 
                                            onclick="event.stopPropagation(); map.setView([${lat}, ${lng}], 17); markers[${index}] && markers[${index}].openPopup();" 
                                            title="Show on Map">
                                        <i class="fas fa-map"></i>
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>`;
        
        container.appendChild(facilityCard);
    });
    
    // Show hospital search filter after facilities are loaded
    if (pageData.length > 0) {
        document.getElementById('hospital-search-section').style.display = 'block';
    }
    
    // Adjust map to show current page markers
    if (pageData.length > 0) {
        const group = L.featureGroup(markers);
        if (userMarker) {
            group.addLayer(userMarker);
        }
        map.fitBounds(group.getBounds().pad(0.1));
    }
}

// Setup pagination controls
function setupPagination() {
    const totalPages = Math.ceil(allFacilitiesData.length / facilitiesPerPage);
    updatePaginationInfo();
}

// Update pagination display
function updatePaginationInfo() {
    const totalPages = Math.ceil(allFacilitiesData.length / facilitiesPerPage);
    const pageInfo = document.getElementById('page-info');
    const prevPage = document.getElementById('prev-page');
    const nextPage = document.getElementById('next-page');
    
    if (pageInfo) {
        pageInfo.textContent = `Page ${currentPage} of ${totalPages}`;
    }
    
    // Update navigation buttons
    if (prevPage) {
        if (currentPage <= 1) {
            prevPage.classList.add('disabled');
        } else {
            prevPage.classList.remove('disabled');
        }
    }
    
    if (nextPage) {
        if (currentPage >= totalPages) {
            nextPage.classList.add('disabled');
        } else {
            nextPage.classList.remove('disabled');
        }
    }
}

// Change page function
function changePage(direction) {
    const totalPages = Math.ceil(allFacilitiesData.length / facilitiesPerPage);
    const newPage = currentPage + direction;
    
    if (newPage >= 1 && newPage <= totalPages) {
        currentPage = newPage;
        displayFacilitiesPage(currentPage);
        updatePaginationInfo();
        
        // Update facility count to show current page info
        updateFacilityCount(allFacilitiesData.length);
    }
}

// Filter hospitals by name
function filterHospitals(searchTerm) {
    const container = document.getElementById('facilities-container');
    const cards = container.querySelectorAll('.facility-card');
    
    searchTerm = searchTerm.toLowerCase().trim();
    let visibleCount = 0;
    
    cards.forEach(card => {
        const hospitalName = card.querySelector('.card-title').textContent.toLowerCase();
        const facilityType = card.querySelector('.text-primary').textContent.toLowerCase();
        
        if (searchTerm === '' || 
            hospitalName.includes(searchTerm) || 
            facilityType.includes(searchTerm)) {
            card.parentElement.style.display = 'block';
            visibleCount++;
        } else {
            card.parentElement.style.display = 'none';
        }
    });
    
    // Update count
    const countText = document.getElementById('count-text');
    const countEl = document.getElementById('facility-count');
    if (countText && countEl) {
        countText.textContent = `${visibleCount} facilities shown`;
        countEl.style.display = visibleCount > 0 ? 'block' : 'none';
    }
}

// Show detailed information about a facility
function showFacilityDetails(placeId, facilityName) {
    const modal = new bootstrap.Modal(document.getElementById('facilityModal'));
    const modalTitle = document.getElementById('facilityModalLabel');
    const modalContent = document.getElementById('facility-details-content');
    
    modalTitle.textContent = facilityName;
    modalContent.innerHTML = `
        <div class="text-center">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-2">Loading facility details...</p>
        </div>
    `;
    
    modal.show();
    
    // Fetch detailed information
    fetch(`/api/facility-details/${placeId}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                modalContent.innerHTML = `
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        Error loading facility details: ${data.error}
                    </div>
                `;
                return;
            }
            
            const facility = data.facility;
            let detailsHtml = `
                <div class="row">
                    <div class="col-12">
                        <h4>${facility.name}</h4>
                    </div>
                </div>
            `;
            
            if (facility.formatted_address) {
                detailsHtml += `
                    <div class="mb-3">
                        <h6><i class="fas fa-map-marker-alt text-primary me-2"></i>Address</h6>
                        <p>${facility.formatted_address}</p>
                    </div>
                `;
            }
            
            if (facility.formatted_phone_number) {
                detailsHtml += `
                    <div class="mb-3">
                        <h6><i class="fas fa-phone text-success me-2"></i>Phone</h6>
                        <p>
                            <a href="tel:${facility.formatted_phone_number}" class="btn btn-success btn-sm">
                                <i class="fas fa-phone me-1"></i>
                                ${facility.formatted_phone_number}
                            </a>
                        </p>
                    </div>
                `;
            }
            
            if (facility.opening_hours) {
                detailsHtml += `
                    <div class="mb-3">
                        <h6><i class="fas fa-clock text-info me-2"></i>Opening Hours</h6>
                        <p>${facility.opening_hours}</p>
                    </div>
                `;
            }
            
            if (facility.website) {
                detailsHtml += `
                    <div class="mb-3">
                        <h6><i class="fas fa-globe text-warning me-2"></i>Website</h6>
                        <p>
                            <a href="${facility.website}" target="_blank" class="btn btn-warning btn-sm">
                                <i class="fas fa-external-link-alt me-1"></i>
                                Visit Website
                            </a>
                        </p>
                    </div>
                `;
            }
            
            modalContent.innerHTML = detailsHtml;
        })
        .catch(error => {
            console.error('Error fetching facility details:', error);
            modalContent.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    Error loading facility details. Please try again.
                </div>
            `;
        });
}

// Calculate route to facility
function calculateRoute(placeId, destLat, destLng) {
    if (!userLocation) {
        alert('Please find your location first.');
        return;
    }
    
    // Open directions in default maps app
    const url = `https://www.google.com/maps/dir/${userLocation.lat},${userLocation.lng}/${destLat},${destLng}`;
    window.open(url, '_blank');
}

// Clear all markers except user marker
function clearMarkers() {
    markers.forEach(marker => {
        map.removeLayer(marker);
    });
    markers = [];
}

// Update facility count display
function updateFacilityCount(count) {
    const countEl = document.getElementById('facility-count');
    const countText = document.getElementById('count-text');
    
    if (countEl && countText) {
        if (count > 0) {
            const totalPages = Math.ceil(count / facilitiesPerPage);
            const startIndex = (currentPage - 1) * facilitiesPerPage + 1;
            const endIndex = Math.min(currentPage * facilitiesPerPage, count);
            
            if (count > facilitiesPerPage) {
                countText.textContent = `${count} facilities found (showing ${startIndex}-${endIndex})`;
            } else {
                countText.textContent = `${count} facilities found`;
            }
            countEl.style.display = 'block';
        } else {
            countEl.style.display = 'none';
        }
    }
}

// Get display name for category
function getCategoryDisplayName(category) {
    const categoryMap = {
        'healthcare.hospital': 'Hospital',
        'healthcare.clinic_or_praxis': 'Clinic',
        'healthcare.pharmacy': 'Pharmacy',
        'healthcare.dentist': 'Dentist',
        'healthcare.physiotherapist': 'Physical Therapy',
        'healthcare.alternative': 'Alternative Medicine',
        'healthcare.veterinary': 'Veterinary'
    };
    
    return categoryMap[category] || 'Healthcare Facility';
}

// Initialize map when modal is shown
function initMapWhenModalShown() {
    console.log('Setting up map modal listener');
    const facilityModal = document.getElementById('facilityFinderModal');
    if (facilityModal) {
        facilityModal.addEventListener('shown.bs.modal', function() {
            console.log('Modal shown, initializing map');
            setTimeout(() => {
                try {
                    if (!map) {
                        initMap();
                    } else {
                        map.invalidateSize();
                    }
                } catch (error) {
                    console.error('Map initialization error:', error);
                }
            }, 200);
        });
        
        facilityModal.addEventListener('hidden.bs.modal', function() {
            console.log('Modal hidden');
            // Clear any active searches
            if (map) {
                clearMarkers();
            }
        });
    }
}

// Call this function when document is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initMapWhenModalShown);
} else {
    initMapWhenModalShown();
}