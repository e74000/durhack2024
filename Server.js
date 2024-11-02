const express = require('express');
const app = express();
const port = 8080;
const path = require('path');

app.get('/', (req, res) => {
    try {
        res.status(200).sendFile(path.join(__dirname + '/placeholder.html'));
    }
    catch (err) {
        console.log(err);
    }
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
    console.log('Server is running on port http://localhost:8080');
}
);
