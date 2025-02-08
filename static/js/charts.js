// Function to update the prediction chart
function updatePredictionChart(data) {
    const traces = [
        {
            x: data.dates,
            y: data.predictions,
            type: 'scatter',
            mode: 'lines',
            name: 'Predicted Sales',
            line: {color: '#2196F3'}
        },
        {
            x: data.dates,
            y: data.lower_bound,
            type: 'scatter',
            mode: 'lines',
            name: 'Lower Bound',
            line: {color: '#90CAF9'},
            fill: 'none'
        },
        {
            x: data.dates,
            y: data.upper_bound,
            type: 'scatter',
            mode: 'lines',
            name: 'Upper Bound',
            line: {color: '#90CAF9'},
            fill: 'tonexty'
        }
    ];
    
    const layout = {
        title: 'Sales Prediction',
        xaxis: {title: 'Date'},
        yaxis: {title: 'Sales Volume'},
        showlegend: true,
        height: 400,
        margin: {t: 50, b: 50, l: 50, r: 50}
    };
    
    Plotly.newPlot('prediction-chart', traces, layout);

    // Calculate and display summary statistics
    const avgPrediction = data.predictions.reduce((a, b) => a + b, 0) / data.predictions.length;
    const maxPrediction = Math.max(...data.predictions);
    const minPrediction = Math.min(...data.predictions);

    const summaryHtml = `
        <div class="col-md-3">
            <div class="card">
                <div class="card-body">
                    <h6 class="card-title">Average Prediction</h6>
                    <p class="card-text">${avgPrediction.toFixed(2)}</p>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card">
                <div class="card-body">
                    <h6 class="card-title">Maximum Prediction</h6>
                    <p class="card-text">${maxPrediction.toFixed(2)}</p>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card">
                <div class="card-body">
                    <h6 class="card-title">Minimum Prediction</h6>
                    <p class="card-text">${minPrediction.toFixed(2)}</p>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card">
                <div class="card-body">
                    <h6 class="card-title">Prediction Range</h6>
                    <p class="card-text">${(maxPrediction - minPrediction).toFixed(2)}</p>
                </div>
            </div>
        </div>
    `;
    document.getElementById('summary-stats').innerHTML = summaryHtml;
}

// Function to update stock trend chart
function updateStockTrendChart(data) {
    const trace1 = {
        x: data.dates,
        y: data.stock,
        type: 'scatter',
        mode: 'lines',
        name: 'Stock Level',
        line: {color: '#4CAF50'}
    };
    
    const trace2 = {
        x: data.dates,
        y: data.sales,
        type: 'scatter',
        mode: 'lines',
        name: 'Sales',
        line: {color: '#FF9800'}
    };
    
    const layout = {
        title: 'Stock vs Sales Trend',
        xaxis: {title: 'Date'},
        yaxis: {title: 'Volume'},
        showlegend: true,
        height: 300
    };
    
    Plotly.newPlot('stock-trend-chart', [trace1, trace2], layout);
}

// Function to update regional performance chart
function updateRegionalChart(data) {
    const regions = Object.keys(data);
    const salesData = regions.map(r => data[r].sales_volume);
    const utilizationData = regions.map(r => data[r].current_utilization * 100);
    
    const trace1 = {
        x: regions,
        y: salesData,
        type: 'bar',
        name: 'Sales Volume',
        marker: {color: '#2196F3'}
    };
    
    const trace2 = {
        x: regions,
        y: utilizationData,
        type: 'bar',
        name: 'Utilization %',
        marker: {color: '#FF9800'},
        yaxis: 'y2'
    };
    
    const layout = {
        title: 'Regional Performance',
        yaxis: {title: 'Sales Volume'},
        yaxis2: {
            title: 'Utilization %',
            overlaying: 'y',
            side: 'right'
        },
        showlegend: true,
        height: 300,
        barmode: 'group'
    };
    
    Plotly.newPlot('regional-performance-chart', [trace1, trace2], layout);
}

// Function to update cost breakdown chart
function updateCostChart(data) {
    const trace = {
        labels: data.labels,
        values: data.values,
        type: 'pie',
        marker: {
            colors: ['#2196F3', '#4CAF50', '#FF9800']
        }
    };
    
    const layout = {
        title: 'Cost Breakdown',
        height: 300,
        showlegend: true
    };
    
    Plotly.newPlot('cost-breakdown-chart', [trace], layout);
}

// Function to update all metrics
function updateMetrics(data) {
    // Update inventory metrics
    document.getElementById('stock-turnover').textContent = data.inventory.stock_turnover.toFixed(2);
    document.getElementById('avg-stock').textContent = data.inventory.avg_stock.toFixed(0);
    document.getElementById('stock-out-days').textContent = data.inventory.stock_out_days;
    
    // Update cost metrics
    document.getElementById('total-cost').textContent = `$${data.cost.total_cost.toFixed(2)}`;
    document.getElementById('unit-cost').textContent = `$${data.cost.unit_cost.toFixed(2)}`;
    document.getElementById('transport-cost').textContent = `$${data.cost.transport_cost.toFixed(2)}`;
    
    // Update utilization metrics
    document.getElementById('avg-utilization').textContent = `${(data.utilization.avg_utilization * 100).toFixed(1)}%`;
    document.getElementById('peak-utilization').textContent = `${(data.utilization.peak_utilization * 100).toFixed(1)}%`;
    document.getElementById('efficiency-rate').textContent = `${(data.utilization.efficiency_rate * 100).toFixed(1)}%`;
}

// Function to update all analytics
function updateAnalytics(data) {
    if (data.status === 'success') {
        updateMetrics(data.data);
        updateStockTrendChart(data.data.charts.stock_trend);
        updateRegionalChart(data.data.charts.regional_performance);
        updateCostChart(data.data.charts.cost_breakdown);
    } else {
        console.error('Error updating analytics:', data.message);
    }
}

// Event listener for form submission
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const loadingSpinner = document.querySelector('.loading-spinner');
    
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Show loading spinner
            loadingSpinner.style.display = 'block';
            
            const formData = new FormData(this);
            
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading spinner
                loadingSpinner.style.display = 'none';
                
                if (data.status === 'success') {
                    updatePredictionChart(data.data);
                    
                    // Update metrics
                    document.getElementById('current-stock').textContent = 
                        data.data.current_metrics.current_stock;
                    document.getElementById('current-utilization').textContent = 
                        (data.data.current_metrics.current_utilization * 100).toFixed(1) + '%';
                    document.getElementById('last-sales').textContent = 
                        data.data.current_metrics.last_actual_sales;
                    
                    // Show results section
                    document.getElementById('results').style.display = 'block';
                } else {
                    alert('Error: ' + data.message);
                }
            })
            .catch(error => {
                // Hide loading spinner
                loadingSpinner.style.display = 'none';
                console.error('Error:', error);
                alert('Error generating predictions');
            });
        });
    }

    // Set default date to today
    const dateInput = document.getElementById('date');
    if (dateInput) {
        const today = new Date();
        const yyyy = today.getFullYear();
        const mm = String(today.getMonth() + 1).padStart(2, '0');
        const dd = String(today.getDate()).padStart(2, '0');
        dateInput.value = `${yyyy}-${mm}-${dd}`;
    }

    // Load initial analytics
    fetch('/get-analytics')
        .then(response => response.json())
        .then(data => updateAnalytics(data))
        .catch(error => console.error('Error loading analytics:', error));
}); 