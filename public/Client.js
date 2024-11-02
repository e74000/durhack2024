async function loadData() {
    try{
        const response = await fetch('http://127.0.0.1:5000/predict-week?starting=1981-01-19');

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        console.log(data);
        //// bit to display data not written yet
    }
    catch (error) {
        console.error('There has been a problem with your fetch operation:', error);
        // display error message 
    }

}

// create event listener for dropdown menu
// will need to have a button to set a date submission thing

function display_data(){
    var data = loadData();
    console.log(data);
    // display data
    // create a variable to write the data to in the html
    // create a table to display data, then append to the body of the page when it exists
    // create a loop to iterate through the data and write it to the table

    data.forEach(function(data) => {
        // write data to table
        // table will have to created here and appended to the body of the page
        // change htmlinner for things with id =  panel[x]

        
    });
}
display_data();


