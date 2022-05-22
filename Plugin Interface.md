## Introduction

With the plugin interface in the HomeSystem you can create your own plugins which allows you to bring
your own features to the system.

## Getting started

1. You have to create a new JavaProject which contains an `plugin.cfg` with the following parameters including:  
   ```yaml
   name: PluginName
   version: Pluginversion (e.g. 1.0)
   main: path to main (e.g. net.test.Main)
   author: Name of the author (optional)
   ```
2. Create your Main class of your plugin which extends `net.enumtype.homesystem.plugin.Plugin`
and implement the needed methods:
   ```java
   package net.test;
   
   import net.enumtype.homesystem.plugin.Plugin;
   
   public class Main extends Plugin {
       
       @Override
       public void start() {
           //Start method of plugin
       }
       
       @Override
       public void stop() {
           //Stop method of plugin
       }
   }
   ```
3. Export your project to a jar **including** the `plugin.cfg`, then move it to your `plugins`
folder and reload/restart the system.

## Listeners

The plugin system also includes an event handling system. Your listener needs to implement the
`net.enumtype.homesystem.plugin.events.Listener`. When you want to have a method called the method
needs an `@EventHandler` annotation. The name of the method does not matter but the parameter of
this method should be the event you want to handle (See [Events](#Events:)):

```java
public class MyListener implements Listener {

    @EventHandler
    public void onClientLogin(ClientLoginEvent e) {
        System.out.println(e.getClient().getName());
    }

}
```

The listener needs to be registered:
```java
package net.test;

import net.enumtype.homesystem.plugin.Plugin;

public class Main extends Plugin {
    
    @Override
    public void start() {
       HomeSystem.getPluginManager().registerListener(new MyListener());
    }
    
    @Override
    public void stop() {
        //Stop method of plugin
    }
}
```

### Events:

In this list are all events listed, which are handled by the system. It is possible that there will
be events added later:
- ClientConnectEvent - when a WebSocket connects to the server
- ClientLoginEvent - when a Client logs in
- ClientLogoutEvent - when a Client logs out
- ClientMessageEvent - when a Client sends a message to the server

## Commands

The plugin system has an implemented command execution, so you can register your own commands which will
be executed when a Client sends this command to the server (You cannot overwrite system commands!). Your
command class needs to implement the `net.enumtype.homesystem.plugin.Command` and the `execure` method to
be able to handle a command execution:
```java
public class MyCommand implements Command {

    @Override
    public void execute(String command, String[] args) {
        
    }
}
```
- `String command` is the name of the command
- `String[] args` are the arguments of the command split by a space

The command needs to be registered as well where the first[^1] argument of `registerCommand` is the command name:
```java
package net.test;

import net.enumtype.homesystem.plugin.Plugin;

public class Main extends Plugin {
    
    @Override
    public void start() {
       HomeSystem.getPluginManager().registerCommand("command", new MyCommand());
    }
    
    @Override
    public void stop() {
        //Stop method of plugin
    }
}
```

[^1]:Command and Listener do not necessary have to be registered in the `start()` method