const WebSocket = require('ws');
const PORT = 5000;
const wsServer = new WebSocket.Server({
    port: PORT
});

wsServer.on('connection', function (socket,request) {
    console.log(request.headers['User-Agent']);
    const header = request.headers['User-Agent'];
    // Some feedback on the console
    console.log("Limit Reached");  
  
    console.log("A client just connected");

    // Attach some behavior to the incoming socket
    socket.on('message', function (msg) {
        console.log("Received message from client: "  + msg);
        // socket.send("Take this back: " + msg);

        // Broadcast that message to all connected clients except sender
        wsServer.clients.forEach(function (client) {
            if (client !== socket) {
                client.send(msg);
            }
        });

      
    });

    socket.on('close', function () {
        console.log('Client disconnected');
    })

  console.log('Client Count: ' + wsServer.clients.size);
});

console.log( (new Date()) + " Server is listening on port " + PORT);