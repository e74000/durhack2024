const express = require('express');
const app = express();
const port = 8080;
const path = require('path');

express.static.mime.define({'text/css': ['css']});
app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (req, res) => {
    try {
        res.status(200).sendFile(path.join(__dirname + '/index.html'));
    }
    catch (err) {
        console.log(err);
        res.status(500).send('Internal Server Error');
    }
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
    console.log('Server is running on port http://localhost:8080');

});
