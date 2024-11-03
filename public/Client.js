// Function to determine the icon based on weather data
function getWeatherIcon(temp, precip, sunshine) {
    if (sunshine > 4 && precip < 1.0) {
        return '<i class="ri-sun-line"></i>';  // Sun
    } else if (sunshine < 2 && precip < 1.0) {
        return '<i class="ri-cloudy-line"></i>';  // Cloudy
    } else if (sunshine < 1 && precip < 0.2) {
        return '<i class="ri-foggy-line"></i>';  // Foggy
    } else if (precip > 2 && precip <= 5 && sunshine < 2) {
        return '<i class="ri-rainy-line"></i>';  // Rainy
    } else if (precip >= 0.5 && precip < 2 && sunshine >= 1 && sunshine < 3) {
        return '<i class="ri-drizzle-line"></i>';  // Drizzle
    } else if (precip > 2 && precip <= 5 && sunshine >= 2 && sunshine < 4) {
        return '<i class="ri-showers-line"></i>';  // Showers
    } else if (precip > 5 && sunshine < 1) {
        return '<i class="ri-heavy-showers-line"></i>';  // Heavy Showers
    } else {
        return '<i class="ri-cloudy-line"></i>';  // Default to Cloudy
    }
}

async function loadData() {
    try {
        const response = await fetch('http://127.0.0.1:5000/predict-week?starting=1999-07-04');

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        displayData(data); // Pass the data to displayData function
    } catch (error) {
        console.error('There has been a problem with your fetch operation:', error);
    }
}

// Function to display data within each tab
function displayData(data) {
    const days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
    days.forEach(day => {
        const panel = document.getElementById(`panel${days.indexOf(day) + 1}`);
        const weatherData = data[day];
        
        if (weatherData) {
            // Set icon based on the heuristic
            const iconHTML = getWeatherIcon(
                weatherData.Prediction_Mean_Temp,
                weatherData.Prediction_Precipitation / 4,
                weatherData.Prediction_Sunshine
            );
            
            // Update the icon and weather info
            panel.querySelector('.weather-icon').innerHTML = iconHTML;
            panel.querySelector('.weather-info').innerHTML = `
                <p><strong>Mean Temperature:</strong> ${weatherData.Prediction_Mean_Temp.toFixed(2)} °C</p>
                <p><strong>Precipitation:</strong> ${(weatherData.Prediction_Precipitation/4).toFixed(2)} mm</p>
                <p><strong>Sunshine:</strong> ${weatherData.Prediction_Sunshine.toFixed(2)} hrs</p>
            `;
        } else {
            panel.querySelector('.weather-info').innerHTML = "<p>No data available for this day.</p>";
        }
    });
}

// Add an event listener to handle tab and panel visibility
function setupTabHighlighting() {
    const navLinks = document.querySelectorAll('.day-tabs .nav-link');
    const panels = document.querySelectorAll('.fade-panel');

    navLinks.forEach(link => {
        link.addEventListener('click', function() {
            // Remove active class from all links and panels
            navLinks.forEach(l => l.classList.remove('active'));
            panels.forEach(panel => panel.classList.remove('active'));

            // Add active class to the clicked link and the corresponding panel
            this.classList.add('active');
            const targetPanel = document.querySelector(this.getAttribute('data-target'));
            targetPanel.classList.add('active');
        });
    });

    // Trigger the first panel to be visible on load
    navLinks[0].click();
}

// Initialize data load and tab highlighting
loadData();
setupTabHighlighting();
