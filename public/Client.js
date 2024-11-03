async function loadData() {
    try{
        const response = await fetch('http://127.0.0.1:5000/predict-week?starting=1981-01-19');

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        console.log('loaded in loadData');
        console.log(data);
        //// bit to display data not written yet
        return data;
    }
    catch (error) {
        console.error('There has been a problem with your fetch operation:', error);
        // display error message 
    }
}
// create event listener for dropdown menu
// will need to have a button to set a date submission thing

async function displayData(){
    let tableHTML = "<table><thead><tr><th>Predicted mean temp</th></tr></thead><tbody>";
    Mon = document.getElementById('Monday');
    loadData().then (function(data){
        console.log('data loaded in display_data');
        console.log(data);
        // create a table
        tableHTML += "<tr><td>" + data.Monday.Prediction_Mean_Temp + "</td></tr>";
        console.log(data.Monday.Prediction_Mean_Temp);
        console.log(data.Monday.Predicted_Mean_Temp);
        // display data
        // create a variable to write the data to in the html
        // create a table to display data, then append to the body of the page when it exists
        // create a loop to iterate through the data and write it to the table

        // create a table

    });

    tableHTML += "</tbody></table>";
    Mon.innerHTML = tableHTML;
    // display data
    // create a variable to write the data to in the html
    // create a table to display data, then append to the body of the page when it exists
    // create a loop to iterate through the data and write it to the table
}
displayData();


