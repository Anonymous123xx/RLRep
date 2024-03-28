const parser = require('@solidity-parser/parser');

const ContractDef = 'ContractDefinition';
const FunctionDef = 'FunctionDefinition';

module.exports.extract = (sourceCode, opts = {}) => {
    const granularity = opts.granularity || 'sourceunit';
    const tolerant = opts.tolerant || false;

    if(granularity == 'sourceunit') {
        return sourceCode;
    }

    const ast = parser.parse(sourceCode, { tolerant, range: true });
    const output = {};

    if(granularity == 'contract') {
        ast.children.forEach(child => {
            if(child.type == ContractDef) {
                const [start, end] = child.range;
                output[child.name] = sourceCode.substring(start, end + 1);
            }
        });
    } else if(granularity == 'function') {
        ast.children.forEach(child => {
            if(child.type == ContractDef) {
                child.subNodes.forEach(grandchild => {
                    if(grandchild.type == FunctionDef) {
                        const [start, end] = grandchild.range;
                        const tmp = sourceCode.substring(start, end + 1);
                        output[child.name] = output[child.name] || {};
                        output[child.name][grandchild.name || 'fallback'] = tmp;
                    }
                });
            }
        });
    }

    return output;
};

module.exports.ParserError = parser.ParserError;
