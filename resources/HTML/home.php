<!DOCTYPE html>
<script>
var ip = location.host;
var ws = new WebSocket("ws://" + ip + ":8000/");
var loggedIn = true;
var user = '';
var loadingDeviceState = [];

ws.onopen = function(e) {
    ws.send('isonline');
};

ws.onmessage = function(e) {
	var data = e.data;
	
	if(data === 'notloggedin') {
		loggedIn = false;
		alert('Please login first!');
		location.replace("/index.php");
	}else if(data === 'loggedin') {
		ws.send('getusername');
		ws.send('xmlrpc getrooms');
	}else if(data.startsWith('user')) {
		user = data.replace('user:', '');
	}else if(data.startsWith('type:rooms')) {
		createRoomElement(data.replace('type:rooms ', ''));
	}else if(data.startsWith('type:roomdevices')) {
		var array = data.replace('type:roomdevices ', '').split(' ');
		var room = array[0];
		var device = array[1];
		
		ws.send('xmlrpc getdevicetype ' + room + ' ' + device);
	}else if(data.startsWith('device')) {
		var array = data.replace('device ', '').split(' ');
		var room = array[0];
		var device = array[1];
		var type = array[2];
		
		createRoomDevice(room, device, type);
	}else if(data.startsWith('type:states')) {
		var array = data.replace('type:states ','').split(' ');
		var device = array[0];
		var state = array[1];
		
		setState(device, state);
	}else if(data.startsWith('noperm')) {
		document.getElementById('room').innerHTML = '';
		document.getElementById('room').setAttribute('class', 'details hide');
		loadingDeviceState = [];
		
		setTimeout(function() {
			alert('No permission for: ' + data.replace('noperm ', ''));
		}, 500);
	}
};

ws.onclose = function(e) {
	ws.close();
	if(loggedIn) {
		loggedIn = false;
		alert('You have been logged out!');
	}
	location.replace("/index.php");
};

ws.onerror = function(e) {
	ws.close();
	loggedIn = false;
	alert('An error has occurred!');
	location.replace("/index.php");
};

function createRoomElement(room) {
	var btn = document.createElement("BUTTON");
	btn.innerHTML = room;
	btn.setAttribute("id", room);
	btn.setAttribute("class", "room");
	
	btn.onclick = function() {
		if(!document.getElementById('room').classList.contains(room)) {
			loadingDeviceState = [];
			if(!document.getElementById('room').classList.contains('hide')) {
				document.getElementById('room').classList.add('hide');
			
				setTimeout(function() {
					ws.send('xmlrpc getdevices ' + room);
					var h1 = document.createElement('H1');
					var text = document.createTextNode(room);
					h1.appendChild(text);
					
					document.getElementById('room').innerHTML = '';
					document.getElementById('room').appendChild(h1);
					document.getElementById('room').setAttribute('class', room);
					loadingDeviceState.push(room);
					loadStates(room);
				}, 500);
			}else {
				ws.send('xmlrpc getdevices ' + room);
				var h1 = document.createElement('H1');
				var text = document.createTextNode(room);
				h1.appendChild(text);
				
				document.getElementById('room').setAttribute('class', room);
				document.getElementById('room').innerHTML = '';
				document.getElementById('room').appendChild(h1);
				loadingDeviceState.push(room);
				loadStates(room);
			}
		}else {
			document.getElementById('room').setAttribute('class', 'detail hide');
		}
	};
	
	document.getElementById("rooms").appendChild(btn);
}

function logout() {
	if(loggedIn) {
		loggedIn = false;
		ws.send('logout ' + user);
		ws.close();
		location.replace("/index.php");
	}
}

function createRoomDevice(room, device, type) {
	if(type === 'ROLL') {
		var container = document.createElement('DIV');
		var p = document.createElement('P');
		var ptext = document.createTextNode(device);
		var input = document.createElement('INPUT');
		var value = document.createElement('SPAN');
		
		container.setAttribute('class', 'devicecontainer');
		container.setAttribute('id', device);
		input.setAttribute('type', 'range');
		input.setAttribute('id', device + ' range');
		value.setAttribute('id', device + ' value');
		value.setAttribute('class', 'rollvalue');
		value.innerHTML = 'Lade...';
		
		p.appendChild(ptext);
		container.appendChild(p);
		container.appendChild(input);
		container.appendChild(value);
		
		input.onchange = function() {
			var state = (parseFloat(input.value) / 100);
			ws.send('xmlrpc setdevice ' + room + ' ' + device + ' ' + state);
		}
		
		document.getElementById('room').appendChild(container);
		document.getElementById('room').classList.remove('hide');
	}else if(type === 'LAMP'){
		var container = document.createElement('DIV');
		var label = document.createElement('LABEL');
		var p = document.createElement('P');
		var ptext = document.createTextNode(device.split('-').join(' '));
		var input = document.createElement('INPUT');
		var span = document.createElement('SPAN');
		
		container.setAttribute('class', 'devicecontainer');
		container.setAttribute('id', device);
		label.setAttribute('class', 'toggle');
		input.setAttribute('type', 'checkbox');
		input.setAttribute('id', device + ' checkbox');
		span.setAttribute('class', 'roundbutton');
		
		p.appendChild(ptext);
		label.appendChild(input);
		label.appendChild(span);
		
		container.appendChild(p);
		container.appendChild(label);
		input.onchange = function() {
			ws.send('xmlrpc setdevice ' + room + ' ' + device + ' ' + input.checked);
		}
		
		document.getElementById('room').appendChild(container);
		document.getElementById('room').classList.remove('hide');
	}
}

function loadStates(room) {
	var interval = setInterval(function() {
		if(loadingDeviceState.includes(room) && loggedIn) {
			var items = document.getElementsByClassName('devicecontainer');
			
			Array.from(items).forEach((item) => {
				ws.send('xmlrpc getdevicestate ' + room + ' ' + item.id);
			});
		}else {
			stopLoadStates(interval);
		}
	}, 1000);
}

function stopLoadStates(interval) {
	clearInterval(interval);
}

function setState(device, stateString) {
	if(stateString.startsWith('STATE')) {
		var state = (stateString.replace('STATE:', '') === 'true');
		document.getElementById(device + ' checkbox').checked = state;
	}else if(stateString.startsWith('LEVEL')) {
		var state = (parseFloat(stateString.replace('LEVEL:', '')) * 100);
		var round = (Math.round(state * 10) / 10)
		
		document.getElementById(device + ' range').value = round.toString();
		document.getElementById(device + ' value').innerHTML = (round.toString() + '%');
	}
}
</script>

<meta content="width=device-width, initial-scale=1" name="viewport" />
<head>
	<title>Home-System � Home</title>
	<link rel="stylesheet" type="text/css" href="style.css">
	<link rel="shortcut icon" href="/favicon.ico">
	<link rel="apple-touch-icon" sizes="180x180" href="/favicon.png">
</head>

<body>
<div class="container">	
	<button id="logout" onclick="logout()">Logout</button>
	<div id="rooms">
	</div><br>
	<div class="detail hide" id="room"></div>
</div>
</body>
