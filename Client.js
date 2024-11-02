function loadData() {
    // Load data from server
    // Display data

    fetch('http://localhost:3000/data')
    .then(response => response.json())
    .then(data => {
        console.log(data);
    });
};

