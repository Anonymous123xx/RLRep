const fs = require('fs');
const path = require('path');

const extractor = require('./extractor');

const [,, ...infiles] = process.argv;

infiles.forEach(file => {
    const addr = path.basename(file, '.sol');
    const code = fs.readFileSync(file, 'utf8');
    try {
        const contracts = extractor.extract(code, { granularity: 'function' });
        Object.entries(contracts).forEach(([contractName, functions]) => {
            Object.entries(functions).forEach(([functionName, functionCode]) => {
                if(functionCode.search("// fault line") != -1) {
                    const fout = fs.createWriteStream(`${addr}.sol`);
                    fout.write(functionCode);
                    fout.end();
                }
            });
        });
    } catch(e) {
        if(e instanceof extractor.ParserError) {
            // TODO
        }
        console.error(e);
    }
});
