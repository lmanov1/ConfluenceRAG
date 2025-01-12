## Wireshark Dissectors Programming Using Lua

### Introduction
Wireshark is a powerful network protocol analyzer that allows users to capture and interactively browse the traffic running on a computer network. Lua is a lightweight, high-level programming language known for its simplicity and speed. Combining these two allows developers to create custom dissectors—modules that interpret and display specific protocol data.

### Setting Up Your Environment
Before you start developing your own Wireshark Lua dissector, you'll need to set up your environment properly.

1. **Install Wireshark**: Download the latest version of Wireshark from the official website.
2. **Install Lua**: Lua comes pre-packaged with Wireshark. Ensure that your version of Lua is compatible with your Wireshark installation.
3. **Check Lua Version**: Verify the installed Lua version using the command line to ensure compatibility.

### Writing Your First Lua Dissector
Understanding the basic structure of a Lua dissector is key to getting started. Here's a simple example of a Lua dissector for a hypothetical protocol:

```lua
-- Simple Wireshark Lua dissector for a custom protocol
local my_proto = Proto("myproto", "My Custom Protocol")

function my_proto.dissector(buffer, pinfo, tree)
    pinfo.cols.protocol = my_proto.name
    local length = buffer:len()
    
    -- Create a display tree
    local subtree = tree:add(my_proto, buffer(), "My Custom Protocol Data")
    
    -- Add fields to the tree
    subtree:add(buffer(0, 1), "Field 1: " .. buffer(0, 1):uint())
    subtree:add(buffer(1, 2), "Field 2: " .. buffer(1, 2):uint())
end

-- Register the dissector for a specific UDP port
local udp_port = DissectorTable.get("udp.port")
udp_port:add(1234, my_proto)
```

### Basic Wireshark Lua API
The Wireshark Lua API provides various functions and classes to help you create dissectors. Here are some basic explanations and examples:

#### 1. **Proto**
The `Proto` class is used to define a new protocol.

```lua
local my_proto = Proto("myproto", "My Custom Protocol")
```

#### 2. **Dissector**
The `dissector` function is where you define how to dissect the protocol data.

```lua
function my_proto.dissector(buffer, pinfo, tree)
    pinfo.cols.protocol = my_proto.name
    local length = buffer:len()
    
    -- Create a display tree
    local subtree = tree:add(my_proto, buffer(), "My Custom Protocol Data")
    
    -- Add fields to the tree
    subtree:add(buffer(0, 1), "Field 1: " .. buffer(0, 1):uint())
    subtree:add(buffer(1, 2), "Field 2: " .. buffer(1, 2):uint())
end
```

#### 3. **Tvb**
The `Tvb` class is used to access the packet data.

```lua
local tvb = buffer()
```

#### 4. **Pinfo**
The `Pinfo` class contains information about the packet.

```lua
local pinfo = pinfo()
```

#### 5. **Tree**
The `Tree` class is used to create the display tree.

```lua
local tree = tree()
```

### Advanced Topics
Once you're comfortable with the basics, you can explore more advanced topics such as TCP reassembly, custom file readers, and handling PDUs in TCP streams. Here are some useful links to get you started:

- [Wireshark Lua/Examples - Wireshark Wiki](https://wiki.wireshark.org/Lua/Examples)
- [Wireshark Lua Dissector: A Quick Start Guide](https://luascripts.com/wireshark-lua-dissector)
- [Example: Dissector written in Lua - Wireshark](https://www.wireshark.org/docs/wsdg_html_chunked/wslua_dissector_example.html)

### Conclusion
Creating Wireshark dissectors using Lua is a powerful way to extend Wireshark's capabilities. With Lua's simplicity and Wireshark's robust framework, you can create custom protocols for packet analysis and gain deeper insights into network traffic.
