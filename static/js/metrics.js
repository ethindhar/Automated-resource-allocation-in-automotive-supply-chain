document.addEventListener('DOMContentLoaded', function() {
    // Fetch metrics data
    fetch('/get-metrics')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                updateMetricsCharts(data.data);
            } else {
                console.error('Error:', data.message);
            }
        })
        .catch(error => console.error('Error:', error));
});

function updateMetricsCharts(data) {
    // Update utilization chart
    const utilizationData = {
        values: [data.resource_utilization * 100],
        gauge: {
            axis: {range: [0, 100]},
            bar: {color: "#2196F3"},
            steps: [
                {range: [0, 30], color: 'lightgray'},
                {range: [30, 70], color: 'gray'},
                {range: [70, 100], color: 'darkgray'}
            ]
        },
        title: {text: "Resource Utilization (%)"},
        type: "indicator",
        mode: "gauge+number"
    };
    
    Plotly.newPlot('utilization-chart', [utilizationData]);
    
    // Add similar charts for velocity and fulfillment
} 