// App State
let allHouseholds = [];
let selectedLclid = null;
let householdProfile = null;
let historicalData = null;

// Chart.js Instances
let forecastChart = null;
let scatterChart = null;
let weekendChart = null;
let timelineChart = null;

// DOM Elements
const searchInput = document.getElementById('search-household');
const selectHousehold = document.getElementById('select-household');
const weatherPreset = document.getElementById('weather-preset');
const customWeatherSliders = document.getElementById('custom-weather-sliders');
const tempSlider = document.getElementById('temp-slider');
const tempVal = document.getElementById('temp-val');
const rangeSlider = document.getElementById('range-slider');
const rangeVal = document.getElementById('range-val');
const btnRunForecast = document.getElementById('btn-run-forecast');

// Profile Elements
const profileId = document.getElementById('profile-id');
const profileArchetype = document.getElementById('profile-archetype');
const profileDemographics = document.getElementById('profile-demographics');
const profileHistoryDays = document.getElementById('profile-history-days');
const metricAvg = document.getElementById('metric-avg');
const metricThermal = document.getElementById('metric-thermal');
const metricWeekend = document.getElementById('metric-weekend');

// Initialize application on load
window.addEventListener('DOMContentLoaded', async () => {
    setupTabListeners();
    setupWeatherControls();
    await loadHouseholds();
    
    // Set initial selection
    if (allHouseholds.length > 0) {
        selectHousehold.value = allHouseholds[0];
        await handleHouseholdChange(allHouseholds[0]);
    }
});

// Setup tab navigation
function setupTabListeners() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            
            tabButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(`tab-${tabId}`).classList.add('active');
            
            // Re-render charts on tab switch to solve layout width issues
            if (tabId === 'insights') {
                renderInsightsCharts();
            } else if (tabId === 'forecasting') {
                if (forecastChart) forecastChart.update();
            }
        });
    });
}

// Setup weather presets and sliders
function setupWeatherControls() {
    weatherPreset.addEventListener('change', () => {
        if (weatherPreset.value === 'custom') {
            customWeatherSliders.classList.remove('hidden');
        } else {
            customWeatherSliders.classList.add('hidden');
        }
    });

    tempSlider.addEventListener('input', () => {
        tempVal.textContent = parseFloat(tempSlider.value).toFixed(1);
    });

    rangeSlider.addEventListener('input', () => {
        rangeVal.textContent = parseFloat(rangeSlider.value).toFixed(1);
    });

    btnRunForecast.addEventListener('click', async () => {
        if (selectedLclid) {
            await runForecast(selectedLclid);
        }
    });
}

// Load household list from backend
async function loadHouseholds() {
    try {
        const res = await fetch('/api/households');
        if (!res.ok) throw new Error("Failed to fetch households list");
        const data = await res.json();
        allHouseholds = data.households;
        
        renderHouseholdList(allHouseholds);
        
        // Setup incremental search
        searchInput.addEventListener('input', () => {
            const query = searchInput.value.toLowerCase().trim();
            const filtered = allHouseholds.filter(id => id.toLowerCase().includes(query));
            renderHouseholdList(filtered);
            
            if (filtered.length > 0) {
                selectHousehold.value = filtered[0];
            }
        });

        selectHousehold.addEventListener('change', async (e) => {
            await handleHouseholdChange(e.target.value);
        });
    } catch (err) {
        console.error("Error loading households:", err);
    }
}

// Render options
function renderHouseholdList(list) {
    selectHousehold.innerHTML = '';
    list.forEach(id => {
        const opt = document.createElement('option');
        opt.value = id;
        opt.textContent = id;
        selectHousehold.appendChild(opt);
    });
}

// Triggered when household is selected
async function handleHouseholdChange(lclid) {
    selectedLclid = lclid;
    profileId.textContent = lclid;
    
    try {
        const [profileRes, historyRes] = await Promise.all([
            fetch(`/api/profile/${lclid}`),
            fetch(`/api/history/${lclid}`)
        ]);
        
        if (!profileRes.ok || !historyRes.ok) throw new Error("Error loading household statistics");
        
        householdProfile = await profileRes.json();
        historicalData = await historyRes.json();
        
        renderProfileInfo();
        await runForecast(lclid);
        
        const activeTab = document.querySelector('.tab-btn.active').getAttribute('data-tab');
        if (activeTab === 'insights') {
            renderInsightsCharts();
        }
    } catch (err) {
        console.error("Error during household change:", err);
    }
}

// Render profile cards in DOM
function renderProfileInfo() {
    const cluster = householdProfile.cluster_id;
    profileArchetype.textContent = householdProfile.cluster_archetype;
    profileArchetype.className = 'badge';
    
    if (cluster === 1) {
        profileArchetype.classList.add('badge-frugal');
    } else if (cluster === 0) {
        profileArchetype.classList.add('badge-thermal');
    } else {
        profileArchetype.classList.add('badge-weekend');
    }
    
    profileDemographics.textContent = householdProfile.acorn_group;
    profileHistoryDays.textContent = `${historicalData.total_records} days`;
    
    metricAvg.textContent = householdProfile.mean_daily_consumption.toFixed(3);
    metricThermal.textContent = householdProfile.thermal_sensitivity.toFixed(2);
    metricWeekend.textContent = householdProfile.weekend_bias.toFixed(2);
}

// Generate simulated weather
function getSimulatedWeather() {
    const preset = weatherPreset.value;
    let avgT = 15.0;
    let rangeT = 6.0;
    
    if (preset === 'winter') {
        avgT = 3.5;
        rangeT = 4.0;
    } else if (preset === 'spring') {
        avgT = 12.0;
        rangeT = 6.0;
    } else if (preset === 'summer') {
        avgT = 23.5;
        rangeT = 8.5;
    } else {
        avgT = parseFloat(tempSlider.value);
        rangeT = parseFloat(rangeSlider.value);
    }
    
    const weather = [];
    const baseDate = new Date(2014, 2, 1);
    
    for (let i = 0; i < 7; i++) {
        const dt = new Date(baseDate);
        dt.setDate(baseDate.getDate() + i);
        
        const dayTemp = avgT + Math.sin(i) * 1.5;
        const dayRange = rangeT + Math.cos(i) * 0.5;
        const hdd = Math.max(0.0, 15.5 - dayTemp);
        const cdd = Math.max(0.0, dayTemp - 22.0);
        const isWeekend = (dt.getDay() === 0 || dt.getDay() === 6) ? 1 : 0;
        const dateStr = dt.toISOString().split('T')[0];
        
        weather.push({
            date: dateStr,
            temp_avg: dayTemp,
            HDD: hdd,
            CDD: cdd,
            temp_range: dayRange,
            is_weekend: isWeekend,
            is_holiday: 0
        });
    }
    
    return weather;
}

// Perform forecast prediction
async function runForecast(lclid) {
    const weatherData = getSimulatedWeather();
    
    try {
        const res = await fetch('/api/forecast', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                lclid: lclid,
                weather: weatherData
            })
        });
        
        if (!res.ok) throw new Error("Inference failed");
        const data = await res.json();
        
        renderWeatherCards(weatherData);
        
        // Manage Anomaly Alert
        const alertBox = document.getElementById('anomaly-alert-box');
        const alertSpan = document.getElementById('anomaly-days');
        if (data.anomalous_days_detected.length > 0) {
            alertBox.classList.remove('hidden');
            const formattedDates = data.anomalous_days_detected.map(d => {
                const dateObj = new Date(d);
                return dateObj.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
            });
            alertSpan.textContent = formattedDates.join(', ');
        } else {
            alertBox.classList.add('hidden');
        }
        
        renderForecastChart(weatherData, data);
        renderRecommendations();
        
    } catch (err) {
        console.error("Forecast error:", err);
    }
}

// Render weather preview cards
function renderWeatherCards(weather) {
    const container = document.getElementById('weather-cards-container');
    container.innerHTML = '';
    
    weather.forEach(w => {
        const card = document.createElement('div');
        card.className = 'weather-card';
        if (w.is_weekend === 1) card.classList.add('weekend');
        
        const dateObj = new Date(w.date);
        const dayLabel = dateObj.toLocaleDateString('en-US', { weekday: 'short' });
        const dateLabel = dateObj.toLocaleDateString('en-US', { month: 'numeric', day: 'numeric' });
        
        card.innerHTML = `
            <div class="weather-card-date">${dayLabel} ${dateLabel}</div>
            <div class="weather-card-temp">${w.temp_avg.toFixed(1)}°C</div>
            <div class="weather-card-type">${w.is_weekend === 1 ? 'Weekend' : 'Weekday'}</div>
        `;
        container.appendChild(card);
    });
}

// Render recommendations
function renderRecommendations() {
    const card = document.getElementById('recs-card');
    const cluster = householdProfile.cluster_id;
    
    card.className = 'recs-card';
    if (cluster === 1) {
        card.classList.add('frugal');
    } else if (cluster === 0) {
        card.classList.add('thermal');
    } else {
        card.classList.add('weekend-recs');
    }
    
    document.getElementById('recs-title').textContent = "Demand Optimization Advisory";
    document.getElementById('recs-obs').textContent = householdProfile.recommendations[0];
    document.getElementById('recs-strategy').textContent = householdProfile.recommendations[1];
    
    const action = householdProfile.recommendations[2].replace('Action Step: ', '');
    document.getElementById('recs-action').textContent = action;
}

// Chart.js Chart render helpers
function renderForecastChart(weather, forecastData) {
    const ctx = document.getElementById('chart-forecast').getContext('2d');
    if (forecastChart) forecastChart.destroy();
    
    const labels = weather.map(w => {
        const dateObj = new Date(w.date);
        return dateObj.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    });
    
    const baselineMean = householdProfile.mean_daily_consumption;
    const thresh = forecastData.anomaly_threshold;
    const dataForecast = forecastData.forecast;
    
    forecastChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Forecasted Demand',
                    data: dataForecast,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.05)',
                    borderWidth: 2,
                    tension: 0.1,
                    pointBackgroundColor: '#10b981',
                    pointRadius: 4,
                    fill: false
                },
                {
                    label: 'Daily Mean Reference',
                    data: Array(7).fill(baselineMean),
                    borderColor: 'rgba(255, 255, 255, 0.15)',
                    borderWidth: 1.5,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false
                },
                {
                    label: 'Anomaly Boundary',
                    data: Array(7).fill(thresh),
                    borderColor: '#ef4444',
                    borderWidth: 1.5,
                    borderDash: [4, 4],
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: '#9ca3af',
                        font: { family: 'Inter', size: 11, weight: '500' }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#6b7280', font: { family: 'Inter', size: 10 } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#6b7280', font: { family: 'Inter', size: 10 } },
                    title: {
                        display: true,
                        text: 'Consumption (kWh)',
                        color: '#6b7280',
                        font: { family: 'Inter', size: 11 }
                    }
                }
            }
        }
    });
}

function renderInsightsCharts() {
    if (!historicalData) return;
    renderScatterChart();
    renderWeekendChart();
    renderTimelineChart();
}

function renderScatterChart() {
    const ctx = document.getElementById('chart-temp-scatter').getContext('2d');
    if (scatterChart) scatterChart.destroy();
    
    const temps = historicalData.scatter_data.temp;
    const energy = historicalData.scatter_data.energy;
    const scatterPoints = temps.map((t, idx) => ({ x: t, y: energy[idx] }));
    
    const slope = historicalData.regression.slope;
    const intercept = historicalData.regression.intercept;
    const minTemp = Math.min(...temps);
    const maxTemp = Math.max(...temps);
    
    const linePoints = [
        { x: minTemp, y: slope * minTemp + intercept },
        { x: maxTemp, y: slope * maxTemp + intercept }
    ];
    
    scatterChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Daily Observation',
                    data: scatterPoints,
                    backgroundColor: 'rgba(156, 163, 175, 0.3)',
                    pointRadius: 3
                },
                {
                    label: 'Regression Line',
                    data: linePoints,
                    type: 'line',
                    borderColor: '#ef4444',
                    borderWidth: 1.5,
                    fill: false,
                    pointRadius: 0,
                    showLine: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#9ca3af', font: { family: 'Inter', size: 11 } }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#6b7280', font: { family: 'Inter', size: 10 } },
                    title: { display: true, text: 'Mean Temperature (°C)', color: '#6b7280', font: { family: 'Inter', size: 11 } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#6b7280', font: { family: 'Inter', size: 10 } },
                    title: { display: true, text: 'Energy (kWh)', color: '#6b7280', font: { family: 'Inter', size: 11 } }
                }
            }
        }
    });
}

function renderWeekendChart() {
    const ctx = document.getElementById('chart-weekend-bar').getContext('2d');
    if (weekendChart) weekendChart.destroy();
    
    const averages = historicalData.day_type_averages;
    
    weekendChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Weekday Average', 'Weekend Average'],
            datasets: [{
                data: [averages.weekday, averages.weekend],
                backgroundColor: ['#576d6b', '#eab308'],
                borderWidth: 0,
                borderRadius: 4,
                barThickness: 40
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: '#9ca3af', font: { family: 'Inter', size: 11 } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#6b7280', font: { family: 'Inter', size: 10 } },
                    title: { display: true, text: 'Energy Load (kWh)', color: '#6b7280', font: { family: 'Inter', size: 11 } }
                }
            }
        }
    });
}

function renderTimelineChart() {
    const ctx = document.getElementById('chart-hist-timeline').getContext('2d');
    if (timelineChart) timelineChart.destroy();
    
    const dates = historicalData.recent_30_days.dates;
    const values = historicalData.recent_30_days.values;
    
    const formattedLabels = dates.map(d => {
        const dateObj = new Date(d);
        return dateObj.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    });
    
    const thresh = householdProfile.anomaly_threshold;
    const anomalyValues = values.map(val => val > thresh ? val : null);
    
    timelineChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: formattedLabels,
            datasets: [
                {
                    label: 'Daily Consumption',
                    data: values,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.02)',
                    borderWidth: 1.5,
                    tension: 0.1,
                    pointRadius: 2,
                    fill: true
                },
                {
                    label: 'Anomaly Flags',
                    data: anomalyValues,
                    borderColor: '#ef4444',
                    backgroundColor: '#ef4444',
                    type: 'scatter',
                    pointRadius: 5,
                    z: 10
                },
                {
                    label: 'Anomaly Boundary',
                    data: Array(dates.length).fill(thresh),
                    borderColor: 'rgba(239, 68, 68, 0.35)',
                    borderWidth: 1,
                    borderDash: [4, 4],
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { color: '#9ca3af', font: { family: 'Inter', size: 11 } }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#6b7280', font: { family: 'Inter', size: 10 } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#6b7280', font: { family: 'Inter', size: 10 } },
                    title: { display: true, text: 'Energy (kWh)', color: '#6b7280', font: { family: 'Inter', size: 11 } }
                }
            }
        }
    });
}
