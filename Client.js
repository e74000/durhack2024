async function loadData() {
    try{
        const repsonse = await fetch('http://localhost:5001/api');

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

};
loadData();

// create event listener for each day of the week

var monday = document.getElementById('monday');
var tuesday = document.getElementById('tuesday');
var wednesday = document.getElementById('wednesday');
var thursday = document.getElementById('thursday');
var friday = document.getElementById('friday');
var saturday = document.getElementById('saturday');
var sunday = document.getElementById('sunday');

monday.addEventListener('click', function() {
    console.log('Monday');
}
);
tuesday.addEventListener('click', function() {
    console.log('Tuesday');
}
);
wednesday.addEventListener('click', function() {
    console.log('Wednesday');
}
);
thursday.addEventListener('click', function() {
    console.log('Thursday');
}
);
friday.addEventListener('click', function() {
    console.log('Friday');
}
);

saturday.addEventListener('click', function() {
    console.log('Saturday');
}
);
sunday.addEventListener('click', function() {
    console.log('Sunday');
}
);

// create event listener for dropdown menu

function display_data(){
    var data = loadData();
    console.log(data);
    // display data
    // create a variable to write the data to in the html
    // create a table to display data, then append to the body of the page when it exists
    // create a loop to iterate through the data and write it to the table

    data.array.forEach(element => {
        
    });





}



