var ip = location.host;
var windowLocation = window.location.pathname;
var fileName = windowLocation.substring(windowLocation.lastIndexOf('/') + 1);
var ws = location.protocol === 'https:' ? new WebSocket("wss://" + ip + ":8001/") : new WebSocket("ws://" + ip + ":8000/");

var username = '';
var activeRoom = '';
var loggedIn = false;

function login() {
    var user = document.getElementById('user').value;
    var pass = document.getElementById('pass').value;
    if(ws.readyState === WebSocket.CLOSED || ws.readyState === WebSocket.CLOSING)
            ws = location.protocol === 'https:' ? new WebSocket("wss://" + ip + ":8001/") : new WebSocket("ws://" + ip + ":8000/");
    ws.send('login ' + user + ' ' + pass);

    return false;
}

function logout() {
    loggedIn = false;
    ws.send('logout ' + username);
    ws.close();
}

function checkLoginCommand(command, args) {
    switch(command) {
        case "verifylogin":
            ws.send('changeconnection');
            ws.close();
            location.replace(windowLocation.substring(0, windowLocation.lastIndexOf("/")) + "/home.html");
            break;
        case "wrongdata":
            document.getElementById('user').value = '';
            document.getElementById('pass').value = '';

            document.getElementById('user').blur();
            document.getElementById('pass').blur();
            alert('Username or password wrong!');
            break;
        default:  console.log('Unknown command ' + command);
    }
}

function checkCommand(command, args) {
    switch(command) {
        case 'username':
            if(args.lenth < 1) return;
            username = args[0];
            loggedIn = true;
            ws.send('xmlrpc getrooms');
            break;
        case 'type:rooms':
            if(args.lenth < 1) return;
            createRoomElement(args[0]);
            break;
        case 'type:roomdevices':
            if(args.lenth < 2) return;
            ws.send('xmlrpc getdevicetype ' + args[0] + ' ' + args[1]);
            break;
        case 'type:states':
            if(args.lenth < 2) return;
            updateState(args[0], args[1].split(':')[0], args[1].split(':')[1]);
            break;
        case 'device':
            if(args.lenth < 3) return;
            createRoomDevice(args[0], args[1], args[2]);
            break;
        case 'noperm':
            if(args.lenth < 1) return;
            document.getElementById('room').innerHTML = '';
            document.getElementById('room').setAttribute('class', 'details hide');
            activeRoom = '';
            setTimeout(() => alert('No permission for: ' + args[0]), 500);
            break;
        default:  console.log('Unknown command ' + command);
    }
}

//Room and device managing

function createRoomElement(room) {
	var btn = document.createElement("BUTTON");
	btn.innerHTML = room;
	btn.setAttribute("id", room);
	btn.setAttribute("class", "room");
	
	btn.onclick = function() {
		if(!document.getElementById('room').classList.contains(room)) {
			ws.send('xmlrpc getdevices ' + room);
			var h1 = document.createElement('H1');
			var text = document.createTextNode(room);
			h1.appendChild(text);
			
			document.getElementById('room').setAttribute('class', room);
			document.getElementById('room').innerHTML = '';
			document.getElementById('room').appendChild(h1);
			activeRoom = room;
			loadStates(room);
		}else {
			document.getElementById('room').setAttribute('class', 'detail hide');
			activeRoom = '';
		}
	};
	
	document.getElementById("rooms").appendChild(btn);
}

function createRoomDevice(room, device, type) {
    var container = document.createElement('DIV');
    var p = document.createElement('P');
    var ptext = document.createTextNode(device.split('-').join(' '));
    var input = document.createElement('INPUT');
    var span = document.createElement('SPAN');
    
    container.setAttribute('class', 'devicecontainer');
    container.setAttribute('id', device);

    switch(type) {
        case 'ROLL':
            input.setAttribute('type', 'range');
            input.setAttribute('id', device + ' range');
            span.setAttribute('id', device + ' value');
            span.setAttribute('class', 'rollvalue');
            span.innerHTML = 'Lade...';
            
            container.appendChild(input);
            container.appendChild(span);
            
            input.onchange = function() {
                var state = (parseFloat(input.value) / 100);
                ws.send('xmlrpc setdevice ' + room + ' ' + device + ' ' + state);
            }
            break;
        case 'LAMP':
            var label = document.createElement('LABEL');

            label.setAttribute('class', 'toggle');
            input.setAttribute('type', 'checkbox');
            input.setAttribute('id', device + ' checkbox');
            span.setAttribute('class', 'roundbutton');
            
            label.appendChild(input);
            label.appendChild(span);
            
            container.appendChild(label);

            input.onchange = function() {
                ws.send('xmlrpc setdevice ' + room + ' ' + device + ' ' + input.checked);
            }
            break;
        default: console.log('Unknown device type ' + type);
    }

    p.appendChild(ptext);
    container.appendChild(p);
    document.getElementById('room').appendChild(container);
    document.getElementById('room').classList.remove('hide');
}

function loadStates(room) {
	var intervalID = setInterval(function() {
		if(activeRoom === room && loggedIn) {
			var items = document.getElementsByClassName('devicecontainer');
            Array.from(items).forEach((item) => ws.send('xmlrpc getdevicestate ' + room + ' ' + item.id));
		}else clearInterval(intervalID);
	}, 1000);
}

function updateState(device, value_key, value) {
    if(value_key === 'WORKING') return;

    switch(value_key) {
        case 'STATE':
            document.getElementById(device + ' checkbox').checked = (value === 'true');
            break;
        case 'LEVEL':
            value = (Math.round((parseFloat(value) * 100) * 10) / 10);
            document.getElementById(device + ' range').value = value.toString();
		    document.getElementById(device + ' value').innerHTML = (value.toString() + '%');
            break;
        default: console.log('Unknown value_key ' + value_key + " for device " + device);
    }
}

//WS Listeners

ws.addEventListener('open', function(evt) {
    var intervalID = setInterval(() => {
        if(ws.readyState === WebSocket.CLOSED || ws.readyState === WebSocket.CLOSING) {
            clearInterval(intervalID);
        }else ws.send('isonline');
    }, 60000);
});

ws.addEventListener('message', function(evt) {
    console.log(evt.data);

    if(evt.data == null) return;
    var data = evt.data.split(' ');
    var command = data[0];
    var args = data.slice(1);

    switch(command) {
        case 'nohttp':
            alert('Websocket does not accept http connection!');
            if(loggedIn) logout;
            return;
        case 'notloggedin':
            if(fileName === 'home.html') {
                alert('Please login at first');
                location.replace(windowLocation.substring(0, windowLocation.lastIndexOf("/")) + "/index.html");
            }
            return;
    }

    switch(fileName) {
        case 'index.html':
            checkLoginCommand(command, args);
            break;
        case 'home.html':
            checkCommand(command, args);
            break;
        default:
            console.log('Unknown filename ' + fileName);
    }
});

ws.addEventListener('close', function(evt) {
    if(loggedIn && fileName === 'home.html') {
        loggedIn = false;
        alert('You have been logged out!');
        location.replace(windowLocation.substring(0, windowLocation.lastIndexOf("/")) + "/index.html");
    }else if(fileName === 'home.html') location.replace(windowLocation.substring(0, windowLocation.lastIndexOf("/")) + "/index.html");
});