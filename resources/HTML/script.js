var ws;

function login() {
    var user = document.getElementById('user').value;
    var pass = document.getElementById('pass').value;
    var ip = location.host;
    ws = location.protocol === 'https:' ? new WebSocket("wss://" + ip + ":8001/") : new WebSocket("ws://" + ip + ":8000/");

    ws.onopen = function(evt) {
        ws.send('login ' + user + ' ' + pass);
    };

    ws.onmessage = function(evt) {
        var data = evt.data;
        if(data === 'verifylogin ' + user) {
            ws.send('changeconnection');
            ws.close();
            var windowLocation = window.location.pathname;
            location.replace(windowLocation.substring(0, windowLocation.lastIndexOf("/")) + "/home.html");
        }else if(data === 'wrongdata') {
            document.getElementById('user').value = '';
            document.getElementById('pass').value = '';

            document.getElementById('user').blur();
            document.getElementById('pass').blur();
            alert('Username or password wrong!');
        }else if(data == 'nohttp') {
            document.getElementById('user').value = '';
            document.getElementById('pass').value = '';

            document.getElementById('user').blur();
            document.getElementById('pass').blur();
            alert('Websocket does not accept http connection!');
        }
    };

    return false;
}

ws.onmessage = function(evt) {
    if(evt.data == null) return;
    var data = evt.data.split(' ');
    var command = data[0];
    var args = data.slice(1);

    switch(command) {
        case "verifylogin":
            break;
        case "wrongdata":
            break;
        case "nohttp":
            break;
        default:
            console.log('Unknown command ' + command);
    }
}