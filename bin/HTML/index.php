<!DOCTYPE html>
<html>
<meta content="width=device-width, initial-scale=1" name="viewport" />
<head>
  <link rel="stylesheet" type="text/css" href="style.css">
  <link rel="shortcut icon" href="/favicon.ico">
  <link rel="apple-touch-icon" sizes="180x180" href="/favicon.png">
  <title>Home-System » Login</title>
</head>

<body>
<div class="container">
<h1> Home-System » Login </h1>
<form id="login"method="post" autocomplete="on" onsubmit="return login();">
<p>Benutzer:</p>
<input type="text" placeholder="Benutzername" id="user" autocomplete="on" required><br>

<p>Passwort:</p>
<input type="password" placeholder="Passwort" id="pass" autocomplete="on" required><br>

<input type="submit" value="Einloggen" id="submit">

<script>
function login() {	
	var user = document.getElementById('user').value;
	var pass = document.getElementById('pass').value;
	var ip = location.host;
	var ws = new WebSocket("ws://" + ip + ":8000/");
	
	ws.onopen = function(evt) {
		ws.send('login ' + user + ' ' + pass);
	};
	
	ws.onmessage = function(evt) {
		var data = evt.data;
		if(data === 'verifylogin ' + user) {
			ws.send('changeconnection');
			closeWS(ws);
			location.replace("/home.php");
		}else if(data === 'wrong data') {
			document.getElementById('user').value = '';
			document.getElementById('pass').value = '';
			
			document.getElementById('user').blur();
			document.getElementById('pass').blur();
			alert('Die eingegebenen Daten sind falsch!');
		}
	};
	
	return false;
}

function closeWS(ws) {
	ws.close();
}
</script>
</form>
</div>
</body>
</html>