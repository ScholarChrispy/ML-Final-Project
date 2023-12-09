$(document).ready(function() {
    var button = document.getElementById("sendButton");
    var textBox = document.getElementById("inputBox");
    button.addEventListener("click", sendString);
    function sendString(){
        var inputString = textBox.value;
        //hi. this function gets run when you click on the button.
        //  it grabs the value in the textbox
    };
})