/**
 * Weather Chart Setup - Production Ready
 * Handles temperature and humidity visualization with comprehensive error handling
 */

(function() {
    'use strict';
    
    // Configuration constants
    const CONFIG = {
        CHART_ID: 'chart',
        SELECTORS: {
            forecast: '.forecast-item',
            time: '.forecast-time',
            temp: '.forecast-temperatureValue',
            humidity: '.forecast-humidityValue',
            wrapper: '.chart-wrapper'
        },
        COLORS: {
            temperature: {
                border: '#ff5733',
                background: 'rgba(255, 87, 51, 0.6)',
                point: '#ffffff'
            },
            humidity: {
                border: '#36a2eb',
                background: 'transparent',
                point: '#ffffff'
            },
            text: '#ffffff',
            grid: 'rgba(255, 255, 255, 0.1)',
            border: 'rgba(255, 255, 255, 0.3)'
        },
        ANIMATION: {
            duration: 800,
            easing: 'easeInOutQuart'
        }
    };

    // Utility functions
    const utils = {
        // Get computed CSS property
        getCSSProperty: (property, fallback = '#ffffff') => {
            try {
                return getComputedStyle(document.documentElement)
                    .getPropertyValue(property).trim() || fallback;
            } catch {
                return fallback;
            }
        },

        // Create gradient
        createGradient: (ctx, colorStops, direction = 'vertical') => {
            try {
                const gradient = direction === 'vertical' 
                    ? ctx.createLinearGradient(0, 0, 0, 200)
                    : ctx.createLinearGradient(0, 0, 200, 0);
                    
                colorStops.forEach(([stop, color]) => {
                    gradient.addColorStop(stop, color);
                });
                return gradient;
            } catch {
                return colorStops[0][1]; // Fallback to first color
            }
        },

        // Debounce function for resize events
        debounce: (func, wait) => {
            let timeout;
            return function executedFunction(...args) {
                const later = () => {
                    clearTimeout(timeout);
                    func(...args);
                };
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
            };
        }
    };

    // Chart error handler
    const handleChartError = (error, chartElement) => {
        console.error('Chart error:', error);
        const wrapper = chartElement.closest(CONFIG.SELECTORS.wrapper);
        if (wrapper) {
            wrapper.innerHTML = `
                <div class="chart-error" style="
                    display: flex; 
                    align-items: center; 
                    justify-content: center; 
                    height: 100%; 
                    color: ${CONFIG.COLORS.text};
                    text-align: center;
                    padding: 2rem;
                ">
                    <div>
                        <p style="margin-bottom: 0.5rem;">Unable to load chart</p>
                        <button onclick="location.reload()" style="
                            background: rgba(255, 255, 255, 0.2);
                            border: 1px solid rgba(255, 255, 255, 0.3);
                            color: white;
                            padding: 0.5rem 1rem;
                            border-radius: 0.5rem;
                            cursor: pointer;
                        ">Refresh Page</button>
                    </div>
                </div>
            `;
        }
    };

    // Extract and validate forecast data
    const extractForecastData = () => {
        try {
            const dataElement = document.getElementById('forecast-data');
            if (!dataElement) {
                throw new Error('Forecast data script element not found');
            }

            const data = JSON.parse(dataElement.textContent);

            // Rename 'hum' to 'humidity' for chart consistency
            const formattedData = data.map(item => ({
                time: item.time,
                temp: item.temp,
                humidity: item.hum
            }));

            if (!Array.isArray(formattedData) || formattedData.length === 0) {
                throw new Error('No valid forecast data found after parsing');
            }

            console.log('Extracted forecast data:', formattedData);
            return formattedData;
        } catch (error) {
            console.error('Data extraction error:', error);
            return [];
        }
    };

    // Create chart configuration
    const createChartConfig = (data, textColor) => {
        const times = data.map(d => d.time);
        const temps = data.map(d => d.temp);
        const humidity = data.map(d => d.humidity);

        return {
            type: 'line',
            data: {
                labels: times,
                datasets: [
                    {
                        label: 'Temperature (°C)',
                        data: temps,
                        borderColor: CONFIG.COLORS.temperature.border,
                        backgroundColor: CONFIG.COLORS.temperature.background,
                        fill: true,
                        borderWidth: 3,
                        tension: 0.4,
                        pointRadius: 5,
                        pointHoverRadius: 8,
                        pointBackgroundColor: CONFIG.COLORS.temperature.point,
                        pointBorderColor: CONFIG.COLORS.temperature.border,
                        pointBorderWidth: 2,
                        yAxisID: 'temperature'
                    },
                    {
                        label: 'Humidity (%)',
                        data: humidity,
                        borderColor: CONFIG.COLORS.humidity.border,
                        backgroundColor: CONFIG.COLORS.humidity.background,
                        fill: false,
                        borderWidth: 3,
                        tension: 0.4,
                        pointRadius: 5,
                        pointHoverRadius: 8,
                        pointBackgroundColor: CONFIG.COLORS.humidity.point,
                        pointBorderColor: CONFIG.COLORS.humidity.border,
                        pointBorderWidth: 2,
                        borderDash: [5, 5],
                        yAxisID: 'humidity'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: textColor,
                            font: { size: 13, weight: '400' },
                            padding: 20,
                            usePointStyle: true,
                            pointStyle: 'circle'
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: CONFIG.COLORS.border,
                        borderWidth: 1,
                        cornerRadius: 8,
                        displayColors: true,
                        callbacks: {
                            label: (context) => {
                                const value = Math.round(context.parsed.y * 10) / 10;
                                return context.dataset.label.includes('Temperature')
                                    ? `Temperature: ${value}°C`
                                    : `Humidity: ${value}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { 
                            color: textColor,
                            font: { size: 12 }
                        },
                        grid: { display: false },
                        border: { color: CONFIG.COLORS.border }
                    },
                    temperature: {
                        type: 'linear',
                        position: 'left',
                        ticks: { 
                            color: textColor,
                            font: { size: 12 },
                            callback: (value) => `${Math.round(value)}°C`
                        },
                        grid: { 
                            color: CONFIG.COLORS.grid,
                            drawOnChartArea: true
                        },
                        border: { color: CONFIG.COLORS.border }
                    },
                    humidity: {
                        type: 'linear',
                        position: 'right',
                        min: 0,
                        max: 100,
                        ticks: { 
                            color: textColor,
                            font: { size: 12 },
                            callback: (value) => `${Math.round(value)}%`
                        },
                        grid: { drawOnChartArea: false },
                        border: { color: CONFIG.COLORS.border }
                    }
                },
                animation: CONFIG.ANIMATION
            }
        };
    };

    // Initialize chart
    const initChart = () => {
        // Check if Chart.js is loaded
        if (!window.Chart) {
            console.error('Chart.js not loaded');
            setTimeout(initChart, 100); // Retry after 100ms
            return;
        }

        const chartElement = document.getElementById(CONFIG.CHART_ID);
        if (!chartElement) {
            console.error('Chart canvas element not found');
            return;
        }

        try {
            const data = extractForecastData();
            if (data.length === 0) {
                handleChartError(new Error('No data available'), chartElement);
                return;
            }

            const ctx = chartElement.getContext('2d');
            if (!ctx) {
                throw new Error('Unable to get 2D context');
            }

            const textColor = utils.getCSSProperty('--text-color');
            const config = createChartConfig(data, textColor);

            // Destroy existing chart
            if (window.weatherChart && typeof window.weatherChart.destroy === 'function') {
                window.weatherChart.destroy();
            }

            // Create new chart
            window.weatherChart = new Chart(ctx, config);
            
            console.log('Chart initialized successfully');

        } catch (error) {
            handleChartError(error, chartElement);
        }
    };

    // Handle window resize
    const handleResize = utils.debounce(() => {
        if (window.weatherChart && typeof window.weatherChart.resize === 'function') {
            try {
                window.weatherChart.resize();
            } catch (error) {
                console.error('Chart resize error:', error);
            }
        }
    }, 250);

    // Initialize when DOM is ready
    const init = () => {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initChart);
        } else {
            initChart();
        }

        // Handle resize
        window.addEventListener('resize', handleResize);

        // Handle visibility change (for performance)
        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState === 'visible' && window.weatherChart) {
                try {
                    window.weatherChart.update('none'); // Update without animation
                } catch (error) {
                    console.error('Chart visibility update error:', error);
                }
            }
        });
    };

    // Start initialization
    init();

    // Expose utilities for debugging (only in development)
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        window.weatherChartDebug = {
            CONFIG,
            utils,
            extractForecastData,
            initChart
        };
    }

})();