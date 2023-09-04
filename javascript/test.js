function test(x, y,z){console.log(x); return [x+x, y+y, z+z]}

function double(x){
    console.log(`js func received ${x}`);
    console.log(`returning '${x} ${x}'`)
    return `${x} ${x}`
}

function delay(ms){return new Promise(resolve => setTimeout(resolve, ms))}

async function fire(){ 
    await delay(100);
        
    console.log(document.getElementById("graph"))
}

document.addEventListener("DOMContentLoaded", async function() {await fire()})