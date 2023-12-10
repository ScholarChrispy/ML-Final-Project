function sentimentSelection(inputArray) {
    const probableSentiment = Math.max(...inputArray);

    if (inputArray.indexOf(probableSentiment) === 0) {
        return 'Positive'
    } else if (inputArray.indexOf(probableSentiment) === 1) {
        return 'Negative'
    } else {
        return 'Neutral'
    }
}

$(document).ready(function() {
    var button = document.getElementById("sendButton");
    var modelType = document.getElementById("modelSelect").value;
    
    // runs when the button is clicked
    button.addEventListener("click", sendToServer);
    
    // sends tweet content and model type to server
    async function sendToServer(){
        var tweetContent = document.getElementById("inputBox").value;

        // build the URL
        // e.g. http://127.0.0.1:5000/GRU/I love you
        url = 'http://127.0.0.1:5000/';  // URL of Python server
        url = url.concat(modelType);
        url = url.concat('/');
        url = url.concat(tweetContent);

        await fetch(url, {
        }).then(async response => {
            // get the data
            const data = await response.json()
            
            // get the sentiment value
            const sentimentValue = sentimentSelection(data.message)

            // update the extension to show the sentiment
            $('#sentimentDisplay').text(sentimentValue)
            $('#sentimentDisplay').show()
        });
    };
})