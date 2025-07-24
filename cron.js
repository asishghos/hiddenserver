import https from "https";
import cron from "cron";

const URL = "https://outfit-styler.onrender.com";

const job = new cron.CronJob('*/10 * * * *', function(){
    https.get(URL, (res) => {
        if(res.statusCode === 200){
            console.log("Request Sent Successfully.");
        }
        else{
            console.log("GET request failed", res.statusCode);
        }
    }).on('error', (e) => {
        console.log("Error while sending cron request", e);
    })
})

export default job;