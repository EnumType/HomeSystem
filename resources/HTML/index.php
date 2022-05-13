<!DOCTYPE html>
<script>
	var salt = "hsnRr4UKMuTsfXHuXQ96NBqJpuJNp49h";
    if(location.protocol !== 'https:') alert("The site is not on https! Your password will be sent unencrypted!");

    function sha512(str) {
        return crypto.subtle.digest("SHA-512", new TextEncoder("utf-8").encode(salt + str)).then(buf => {
            return Array.prototype.map.call(new Uint8Array(buf), x=>(('00'+x.toString(16)).slice(-2))).join('');
        });
    }

    function login() {
        var user = document.getElementById('user').value;
        var pass = document.getElementById('pass').value;
        //var ip = location.host;
        var ip = "192.168.2.3";
        var ws = new WebSocket("ws://" + ip + ":8000/");

        if(location.protocol === 'https:') {
            sha512(pass).then(x => {
                ws.onopen = function(evt) {
                    ws.send('login ' + user + ' ' + x);
                };
            });
        }else {
            ws.onopen = function(evt) {
                ws.send('login ' + user + ' ' + pass);
            };
        }

        ws.onmessage = function(evt) {
            var data = evt.data;
            if(data === 'verifylogin ' + user) {
                ws.send('changeconnection');
                closeWS(ws);
                var windowLocation = window.location.pathname;
                location.replace(windowLocation.substring(0, windowLocation.lastIndexOf("/")) + "/home.php");
            }else if(data === 'wrongdata') {
                document.getElementById('user').value = '';
                document.getElementById('pass').value = '';

                document.getElementById('user').blur();
                document.getElementById('pass').blur();
                alert('Username or password wrong!');
            }
        };

        return false;
    }

    function closeWS(ws) {
        ws.close();
    }
</script>
<html>
<meta content="width=device-width, initial-scale=1" name="viewport" />
<head>
  <link rel="stylesheet" type="text/css" href="style.css">
  <link rel="shortcut icon" href="./favicon.ico">
  <link rel="apple-touch-icon" sizes="180x180" href="./favicon.png">
  <title>Home-System >> Login</title>
</head>

<body>
<div class="container">
<h1> Home-System >> Login </h1>
<form id="login"method="post" autocomplete="on" onsubmit="return login();">
<p>Username:</p>
<input type="text" placeholder="Username" id="user" autocomplete="on" required><br>

<p>Password:</p>
<input type="password" placeholder="Password" id="pass" autocomplete="on" required><br>

<input type="submit" value="Login" id="submit">
</form>
</div>
</body>
</html>
