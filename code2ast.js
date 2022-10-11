'use strict';
var parser = require('solidity-parser-antlr');
var fs = require('fs');
var path = require('path');

var type0 = ['SourceUnit', 'InheritanceSpecifier', 'ElementaryTypeName', 'FunctionCall', 'ReturnStatement', 'Block', 'ExpressionStatement', 'BinaryOperation', 'ElementaryTypeNameExpression', 'EmitStatement', 'UnaryOperation', 'StateVariableDeclaration', 'IndexAccess', 'NewExpression', 'VariableDeclarationStatement', 'ArrayTypeName', 'AssemblyAssignment', 'AssemblyBlock', 'InlineAssemblyStatement', 'ForStatement', 'IfStatement', 'Mapping', 'TupleExpression', 'ThrowStatement', 'Conditional', 'AssemblyLocalDefinition', 'BreakStatement', 'ContinueStatement', 'AssemblyCase', 'AssemblySwitch', 'WhileStatement', 'AssemblyFor', 'AssemblyIf', 'FunctionTypeName', 'DoWhileStatement'];
var type1 = ['PragmaDirective', 'StringLiteral', 'BooleanLiteral', 'HexNumber', 'DecimalNumber', 'HexLiteral']; // type, value
var type2 = ['ImportDirective']; // type, path
var type3 = ['ContractDefinition']; // kind, name
var type4 = ['UserDefinedTypeName']; // type, namePath
var type5 = ['UsingForDeclaration']; // Library, libraryName
var type6 = ['VariableDeclaration', 'Identifier', 'FunctionDefinition', 'EventDefinition', 'ModifierDefinition', 'ModifierInvocation', 'EnumValue', 'EnumDefinition', 'StructDefinition', 'LabelDefinition', 'AssemblyFunctionDefinition']; // type, name
var type7 = ['MemberAccess']; // type, memberName
var type8 = ['NumberLiteral']; // type, number
var type9 = ['AssemblyCall']; // type, functionName

var initial_identifier_dict = fs.readFileSync('top300_identifier_dict.txt');
initial_identifier_dict = initial_identifier_dict.toString();
initial_identifier_dict = JSON.parse(initial_identifier_dict);

var len_top300_per_token_dict = new Array();
for(var key in initial_identifier_dict){
	len_top300_per_token_dict[key] = Object.keys(initial_identifier_dict[key]).length;
}

var filePath = path.resolve('dataset_vul/newALLBUGS/validation/function2/');
var count_index = 1;
fs.readdir(filePath,function(err,files){
	if(err){
		console.warn(err)
	}else{
		files.forEach(function(filename){
			console.log("processing " + count_index++ + "-th file : " + filename);
			var filedir = path.join(filePath,filename);
			var address = filename.replace(/\.sol/g,"");
			var data = fs.readFileSync(filedir);
			var code = data.toString();
			var processedCode = code.replace(/&#39;/g,"\'");
			try{
				var result = parser.parse(processedCode,{loc : true});
			}catch(err){
				console.error('Failed to parse, ' + address + '\n',err);
    			return;
			}

			global.identifier_dict = JSON.parse(JSON.stringify(initial_identifier_dict))
			preOrderTraversel(result);

			var tokenSequences = tokenizeSourceUnit(result);
			fs.writeFileSync('dataset_vul/newALLBUGS/validation/ast/' + address + '.sol', '');
			for(var i=0; i < tokenSequences.length; i++){
				fs.appendFileSync('dataset_vul/newALLBUGS/validation/ast/' + address + '.sol', tokenSequences[i] + '\r\n');
			}
		});
	}
});


function preOrderTraversel(root){
	if(typeof root == 'string' || typeof root == 'number' || root == null){
		return;
	}
	if(root instanceof Array){
		// console.log();
	}else{
		if(root.hasOwnProperty('type')){
			extractIdentifer(root);
		}
	}
	for(var key in root){
		preOrderTraversel(root[key]);
	}
}

function extractIdentifer(_AST){
	if(type1.indexOf(_AST.type) > -1){
		pushIdentifierDict(_AST.type, _AST.value);
		return;
	}

	if(type2.indexOf(_AST.type) > -1){
		pushIdentifierDict(_AST.type, _AST.path);
		return;
	}

	if(type3.indexOf(_AST.type) > -1){
		pushIdentifierDict(_AST.kind, _AST.name);
		return;
	}
	if(type4.indexOf(_AST.type) > -1){
		pushIdentifierDict(_AST.type, _AST.namePath);
		return;
	}
	if(type5.indexOf(_AST.type) > -1){
		pushIdentifierDict('Library', _AST.libraryName);
		return;
	}
	if(type6.indexOf(_AST.type) > -1){
		pushIdentifierDict(_AST.type, _AST.name);
		return;
	}
	if(type7.indexOf(_AST.type) > -1){
		pushIdentifierDict(_AST.type, _AST.memberName);
		return;
	}
	if(type8.indexOf(_AST.type) > -1){
		pushIdentifierDict(_AST.type, _AST.number);
		return;
	}
	if(type9.indexOf(_AST.type) > -1){
		pushIdentifierDict(_AST.type, _AST.functionName);
		return;
	}
}

function pushIdentifierDict(key, value){
	if(!global.identifier_dict[key].hasOwnProperty(value)){
		global.identifier_dict[key][value] = key + '_' + (Object.keys(global.identifier_dict[key]).length - len_top300_per_token_dict[key]).toString();
	}
}


function tokenizeSourceUnit(_AST){
	var tokenSequences = [];
	if(_AST.children.length != 0){
		for(var i = 0; i < _AST.children.length; i++){
			switch(_AST.children[i].type){
    			case 'PragmaDirective':
        			break;
    			case 'ContractDefinition':
        			tokenSequences.push(tokenizeContractDefinition(_AST.children[i]));
        			break;
        		case 'ImportDirective':
        			break;
    			default:
    				throw "error";
			}
		}
	}
	return tokenSequences;
}

function tokenizePragmaDirective(_AST){
	return ('pragma' + ' ' + _AST.name + ' ' + global.identifier_dict[_AST.type][_AST.value] + ' ' + ';');
}

function tokenizeContractDefinition(_AST){
	var tokenSequence = '';
	tokenSequence += (_AST.kind + ' ' + global.identifier_dict[_AST.kind][_AST.name]);
	if(_AST.baseContracts.length != 0){
		tokenSequence += (' ' + 'is');
		for(var i = 0; i < _AST.baseContracts.length; i++){
			if(i != 0){
				tokenSequence += (' ' + ',');
			}
			switch(_AST.baseContracts[i].type){
				case 'InheritanceSpecifier':
					tokenSequence += tokenizeInheritanceSpecifier(_AST.baseContracts[i]);
					break;
				default:
					throw "error";
			}
		}
	}
	tokenSequence += (' ' + '{\n');
	if(_AST.subNodes.length != 0){
		for(var i = 0; i < _AST.subNodes.length ; i++){
			switch(_AST.subNodes[i].type){
				case 'FunctionDefinition':
					tokenSequence += (tokenizeFunctionDefinition(_AST.subNodes[i]) + '\n');
					break;
				case 'EventDefinition':
					tokenSequence += (tokenizeEventDefinition(_AST.subNodes[i]) + '\n');
					break;
				case 'StateVariableDeclaration':
					break;
				case 'ModifierDefinition':
					break;
				case 'StructDefinition':
					break;
				case 'UsingForDeclaration':
					break;
				case 'EnumDefinition':
					break;
				default:
    				throw "error";
			}
		}
	}
	tokenSequence += ('}');
	return tokenSequence;
}

function tokenizeEventDefinition(_AST){
	var tokenSequence = '';
	tokenSequence += ('event' + ' ' + global.identifier_dict[_AST.type][_AST.name] + ' ' + '(');
	if(_AST.parameters.length != 0){
		tokenSequence += tokenizeParameters(_AST.parameters);
	}
	tokenSequence += (' ' + ')');
	if(_AST.isAnonymous != false){
		tokenSequence += (' ' + 'anonymous');
	}
	tokenSequence += (' ' + ';');
	return tokenSequence;
}

function tokenizeStateVariableDeclaration(_AST){
	var tokenSequence = '';
	for(var i =0; i < _AST.variables.length; i++){
		if(_AST.variables[i] != null){
			switch(_AST.variables[i].type){
				case 'VariableDeclaration':
					tokenSequence += tokenizeVariableDeclaration(_AST.variables[i]);
					break;
				default:
					throw "error";
			}
		}
	}
	return tokenSequence;
}

function tokenizeModifierDefinition(_AST){
	var tokenSequence = '';
	tokenSequence += (' ' + 'modifier' + ' ' + global.identifier_dict[_AST.type][_AST.name] + ' ' + '(');
	if(_AST.parameters != null){
		if(_AST.parameters.length != 0){
			tokenSequence += tokenizeParameters(_AST.parameters);
		}
	}
	tokenSequence += (' ' + ')');
	if(_AST.body != null){
		tokenSequence += tokenizeBlock(_AST.body);
	}
	return tokenSequence;
}

function tokenizeStructDefinition(_AST){
	var tokenSequence = '';
	tokenSequence += (' ' + 'struct' + ' ' + global.identifier_dict[_AST.type][_AST.name] + ' ' + '{');
	for(var i =0; i < _AST.members.length; i++){
		if(i != 0){
			tokenSequence += (' ' + ';');
		}
		switch(_AST.members[i].type){
			case 'VariableDeclaration':
				tokenSequence +=tokenizeVariableDeclaration(_AST.members[i]);
				break;
			default:
				throw "error";
		}
	}
	tokenSequence += (' ' + '}');
	return tokenSequence;
}

function tokenizeUsingForDeclaration(_AST){
	var tokenSequence = '';
	tokenSequence += (' ' + 'using');
	tokenSequence += (' ' + identifier_dict['Library'][_AST.libraryName]);
	tokenSequence += (' ' + 'for');
	if(_AST.typeName != null){
		tokenSequence += tokenizeTypeName(_AST.typeName);
	}
	return tokenSequence;
}

function tokenizeEnumDefinition(_AST){
	var tokenSequence = '';
	tokenSequence += (' ' + 'enum' + ' ' + global.identifier_dict[_AST.type][_AST.name] + ' ' + '{');
	for(var i =0; i < _AST.members.length; i++){
		if(i != 0){
			tokenSequence += (' ' + ',');
		}
		switch(_AST.members[i].type){
			case 'EnumValue':
				tokenSequence += tokenizeEnumValue(_AST.members[i]);
				break;
			default:
				throw "error";
		}
	}
	tokenSequence += (' ' + '}');
	return tokenSequence;
}

function tokenizeFunctionDefinition(_AST){
	var tokenSequence = '';
	tokenSequence += ('function' + ' ' + global.identifier_dict[_AST.type][_AST.name] + ' ' + '(');
	if(_AST.parameters.length != 0){
		tokenSequence += tokenizeParameters(_AST.parameters);
	}
    tokenSequence += (' ' + ')' );
    if(_AST.visibility != null){
    	tokenSequence += (' ' + _AST.visibility);
    }
    if(_AST.stateMutability != null){
    	tokenSequence += (' ' + _AST.stateMutability);
    }
    if(_AST.modifiers.length != 0){
    	for(var i = 0; i < _AST.modifiers.length; i++){
    		tokenSequence += tokenizeModifierInvocation(_AST.modifiers[i]);
    	}
	}
	if(_AST.returnParameters != null){
		if(_AST.returnParameters.length != 0){
    		tokenSequence += (' ' + 'returns' + ' ' + '(');
			tokenSequence += tokenizeParameters(_AST.returnParameters);
			tokenSequence += (' ' + ')');
		}
	}
    if(_AST.body != null){
    	tokenSequence += tokenizeBlock(_AST.body);
    }else{
    	tokenSequence += (' ' + ';');
    }
    return tokenSequence;
}

function tokenizeParameters(_AST){
	var tokenSequence = '';
	for(var i = 0; i < _AST.length; i++){
		if(i != 0){
			tokenSequence += (' ' + ',');
		}
		switch(_AST[i].type){
			case 'VariableDeclaration':
				tokenSequence += tokenizeVariableDeclaration(_AST[i]);
				break;
			case 'Identifier':
				tokenSequence += tokenizeIdentifier(_AST[i]);
				break;
			default:
				throw "error";
		}
	}
	return tokenSequence;
}

function tokenizeVariableDeclaration(_AST){
	var tokenSequence = '';
	if(_AST.typeName != null){
		tokenSequence += tokenizeTypeName(_AST.typeName);
	}
	if(_AST.isIndexed != false){
		tokenSequence += (' ' + 'indexed');
	}
	if(_AST.visibility != null & _AST.visibility != 'default'){
		tokenSequence += (' ' + _AST.visibility);
	}
	if(_AST.isDeclaredConst == true){
		tokenSequence += (' ' + 'constant');
	}
	if(_AST.name != null){
		tokenSequence += (' ' + global.identifier_dict[_AST.type][_AST.name]);
	}
	if(_AST.expression != null){
		tokenSequence += (' ' + '=');
		tokenSequence += tokenizeExpression(_AST.expression);
	}

	return tokenSequence;
}

function tokenizeArrayTypeName(_AST){
	var tokenSequence = '';
	if(_AST.baseTypeName != null){
		tokenSequence += tokenizeTypeName(_AST.baseTypeName);
	}
	tokenSequence += (' ' + '[');
	if(_AST.length != null){
		switch(_AST.length.type){
			case 'NumberLiteral':
				tokenSequence += tokenizeNumberLiteral(_AST.length);
				break;
			case 'Identifier':
				tokenSequence += tokenizeIdentifier(_AST.length);
				break;
			case 'BinaryOperation':
				tokenSequence += tokenizeBinaryOperation(_AST.length);
				break;
			default:
				throw "error";
		}
	}
	tokenSequence += (' ' + ']');
	return tokenSequence;
}

function tokenizeElementaryTypeName(_AST){
	return (' ' + _AST.name);
}

function tokenizeNumberLiteral(_AST){
	var tokenSequence = '';
	tokenSequence += (' ' + global.identifier_dict[_AST.type][_AST.number]);
	if(_AST.subdenomination != null){
		tokenSequence += (' ' + _AST.subdenomination);
	}
	return tokenSequence;
}


function tokenizeBlock(_AST){
	var tokenSequence = '';
	tokenSequence += (' ' + '{');
	if(_AST.statements != null){
		for(var i = 0; i < _AST.statements.length; i++){
			tokenSequence += tokenizeStatement(_AST.statements[i]);
		}
	}
	tokenSequence += (' ' + '}');
	return tokenSequence;
}

function tokenizeStatement(_AST){
	var tokenSequence = '';
	switch(_AST.type){
		case 'VariableDeclarationStatement':
			tokenSequence += tokenizeVariableDeclarationStatement(_AST);
			tokenSequence += (' ' + ';');
			break;
		case 'ExpressionStatement':
			tokenSequence += tokenizeExpressionStatement(_AST);
			tokenSequence += (' ' + ';');
			break;
		case 'IfStatement':
			tokenSequence += tokenizeIfStatement(_AST);
			break;
		case 'ReturnStatement':
			tokenSequence += tokenizeReturnStatement(_AST);
			tokenSequence += (' ' + ';');
			break;
		case 'ForStatement':
			tokenSequence += tokenizeForStatement(_AST);
			break;
		case 'WhileStatement':
			tokenSequence += tokenizeWhileStatement(_AST);
			break;
		case 'EmitStatement':
			tokenSequence += tokenizeEmitStatement(_AST);
			tokenSequence += (' ' + ';');
			break;
		case 'InlineAssemblyStatement':
			tokenSequence += tokenizeInlineAssemblyStatement(_AST);
			tokenSequence += (' ' + ';');
			break;
		case 'ContinueStatement':
			tokenSequence += tokenizeContinueStatement(_AST);
			tokenSequence += (' ' + ';');
			break;
		case 'BreakStatement':
			tokenSequence += tokenizeBreakStatement(_AST);
			tokenSequence += (' ' + ';');
			break;
		case 'ThrowStatement':
			tokenSequence += tokenizeThrowStatement(_AST);
			tokenSequence += (' ' + ';');
			break;
		case 'Block':
			tokenSequence += tokenizeBlock(_AST);
			break;
		case 'DoWhileStatement':
			tokenSequence += tokenizeDoWhileStatement(_AST);
			tokenSequence += (' ' + ';');
			break;
		default:
			throw "error";
    }
	return tokenSequence;
}


function tokenizeExpressionStatement(_AST){
	var tokenSequence = '';
	if(_AST.expression != null){
		tokenSequence += tokenizeExpression(_AST.expression);
	}
	return tokenSequence;
}
function tokenizeExpression(_AST){
	if(_AST !=  null){
		switch(_AST.type){
			case 'FunctionCall':
				return tokenizeFunctionCall(_AST);
			case 'BinaryOperation':
				return tokenizeBinaryOperation(_AST);
			case 'UnaryOperation':
				return tokenizeUnaryOperation(_AST);
			case 'Identifier':
				return tokenizeIdentifier(_AST);
			case 'IndexAccess':
				return tokenizeIndexAccess(_AST);
			case 'MemberAccess':
				return tokenizeMemberAccess(_AST);
			case 'ElementaryTypeNameExpression':
				return tokenizeElementaryTypeNameExpression(_AST);
			case 'NewExpression':
				return tokenizeNewExpression(_AST);
			case 'TupleExpression':
				return tokenizeTupleExpression(_AST);
			case 'BooleanLiteral':
				return tokenizeBooleanLiteral(_AST);
			case 'NumberLiteral':
				return tokenizeNumberLiteral(_AST);
			case 'StringLiteral':
				return tokenizeStringLiteral(_AST);
			case 'UnaryOperation':
				return tokenizeUnaryOperation(_AST);
			case 'TupleExpression':
				return tokenizeTupleExpression(_AST);
			case 'Conditional':
				return tokenizeConditional(_AST);
			case 'HexNumber':
				return tokenizeHexNumber(_AST);
			case 'DecimalNumber':
				return tokenizeDecimalNumber(_AST);
			case 'AssemblyCall':
				return tokenizeAssemblyCall(_AST);
			case 'HexLiteral':
				return tokenizeHexLiteral(_AST);
			default:
				throw "error";
		}
	}else{
		return '';
	}

}

function tokenizeIfStatement(_AST){
	var tokenSequence = '';
	tokenSequence += (' ' + 'If' + ' ' + '(');
	if(_AST.condition != null){
		tokenSequence += tokenizeCondition(_AST.condition);
	}
	tokenSequence += (')');
	if(_AST.trueBody.type == 'Block'){
		tokenSequence += tokenizeBlock(_AST.trueBody);
	}else{
		tokenSequence += tokenizeStatement(_AST.trueBody);
	}
	if(_AST.falseBody != null){
		tokenSequence += (' ' + 'else');
		if(_AST.falseBody.type == 'Block'){
			tokenSequence += tokenizeBlock(_AST.falseBody);
		}else{
			tokenSequence += tokenizeBlock(_AST.falseBody);
		}
	}
	return tokenSequence;
}

function tokenizeReturnStatement(_AST){
	var tokenSequence = '';
	tokenSequence += (' ' + 'return');
	if(_AST.expression != null){
		tokenSequence += tokenizeExpression(_AST.expression);
	}
	return tokenSequence;
}

function tokenizeForStatement(_AST){
	var tokenSequence = '';
	tokenSequence += (' ' + 'For' + ' ' + '(');
	if(_AST.initExpression != null){
		switch(_AST.initExpression.type){
			case 'VariableDeclarationStatement':
				tokenSequence += tokenizeVariableDeclarationStatement(_AST.initExpression);
				break;
			case 'ExpressionStatement':
				tokenSequence += tokenizeExpressionStatement(_AST.initExpression);
				break;
			default:
				throw "error";
		}
	}
	if(_AST.conditionExpression != null){
		tokenSequence += (' ' + ';');
		tokenSequence += tokenizeCondition(_AST.conditionExpression);
	}
	if(_AST.loopExpression != null){
		tokenSequence += (' ' + ';');
		switch(_AST.loopExpression.type){
			case 'ExpressionStatement':
				tokenSequence += tokenizeExpressionStatement(_AST.loopExpression);
				break;
			default:
				throw "error";
		}
	}
	if(_AST.body.type == 'Block'){
		tokenSequence += tokenizeBlock(_AST.body);
	}else{
		tokenSequence += tokenizeStatement(_AST.body);
	}
	return tokenSequence;
}

function tokenizeWhileStatement(_AST){
	var tokenSequence = '';
	tokenSequence += (' ' + 'While' + ' ' + '(');
	if(_AST.condition != null){
		tokenSequence += tokenizeCondition(_AST.condition);
	}
	tokenSequence += (')');
	if(_AST.body.type == 'Block'){
		tokenSequence += tokenizeBlock(_AST.body);
	}else{
		tokenSequence += tokenizeStatement(_AST.body);
	}
	return tokenSequence;
}

function tokenizeEmitStatement(_AST){
	var tokenSequence = '';
	tokenSequence += (' ' + 'emit');
	switch(_AST.eventCall.type){
		case 'FunctionCall':
			tokenSequence += tokenizeFunctionCall(_AST.eventCall);
			break;
		default:
			throw "error";
	}
	return tokenSequence;
}

function tokenizeInlineAssemblyStatement(_AST){
	var tokenSequence = '';
	tokenSequence += (' ' + 'assembly');
	if(_AST.body != null){
		switch(_AST.body.type){
			case 'AssemblyBlock':
				tokenSequence += tokenizeAssemblyBlock(_AST.body);
				break;
			default:
				throw "error";
		}
	}
	return tokenSequence;
}

function tokenizeContinueStatement(_AST){
	return (' ' + 'continue' + ' ' + ';');
}

function tokenizeBreakStatement(_AST){
	return (' ' + 'break' + ' ' + ';');
}

function tokenizeThrowStatement(_AST){
	return (' ' + 'throw' + ' ' + ';');
}

function tokenizeArguments(_AST){
	var tokenSequence = '';
	for(var i = 0; i < _AST.length; i++){
		if(i != 0){
			tokenSequence += (' ' + ',');
		}
		tokenSequence += tokenizeExpression(_AST[i]);
	}
	return tokenSequence;
}

function tokenizeFunctionCall(_AST){
	var tokenSequence = '';
	tokenSequence += tokenizeExpression(_AST.expression);
	tokenSequence += (' ' + '(');
	if(_AST.arguments.length != 0){
		tokenSequence += tokenizeArguments(_AST.arguments);
	}
	tokenSequence += (' ' + ')');
	return tokenSequence;
}

function tokenizeBinaryOperation(_AST){
	var tokenSequence = '';
	if(_AST.left != null){
		tokenSequence += tokenizeExpression(_AST.left);
	}
	tokenSequence += (' ' + _AST.operator);
	if(_AST.right != null){
		tokenSequence += tokenizeExpression(_AST.right);
	}
	return tokenSequence;
}

function tokenizeUnaryOperation(_AST){
	var tokenSequence = '';
	if(_AST.isPrefix){
		tokenSequence += (' ' + _AST.operator);
		if(_AST.subExpression != null){
			tokenSequence += tokenizeExpression(_AST.subExpression);
		}
	}else{
		if(_AST.subExpression != null){
			tokenSequence += tokenizeExpression(_AST.subExpression);
		}
		tokenSequence += (' ' + _AST.operator);
	}
	return tokenSequence;
}

function tokenizeIdentifier(_AST){
	return (' ' + global.identifier_dict[_AST.type][_AST.name]);
}

function tokenizeIndexAccess(_AST){
	var tokenSequence = '';
	if(_AST.base != null){
		tokenSequence += tokenizeExpression(_AST.base);
	}
	tokenSequence += (' ' + '[');
	if(_AST.index != null){
		tokenSequence += tokenizeExpression(_AST.index);
	}
	tokenSequence += (' ' + ']');
	return tokenSequence;
}

function tokenizeCondition(_AST){
	return tokenizeExpression(_AST);
}

function tokenizeAssemblyBlock(_AST){
	var tokenSequence = '';
	tokenSequence += (' ' + '{');
	for(var i=0; i < _AST.operations.length; i++){
		switch(_AST.operations[i].type){
			case 'AssemblyAssignment':
				tokenSequence += tokenizeAssemblyAssignment(_AST.operations[i]);
				break;
			case 'AssemblyLocalDefinition':
				tokenSequence += tokenizeAssemblyLocalDefinition(_AST.operations[i]);
				break;
			case 'AssemblyCall':
				tokenSequence += tokenizeAssemblyCall(_AST.operations[i]);
				break;
			case 'AssemblySwitch':
				tokenSequence += tokenizeAssemblySwitch(_AST.operations[i]);
				break;
			case 'AssemblyIf':
				tokenSequence += tokenizeAssemblyIf(_AST.operations[i]);
				break;
			case 'LabelDefinition':
				tokenSequence += tokenizeLabelDefinition(_AST.operations[i]);
				break;
			case 'AssemblyFunctionDefinition':
				tokenSequence += tokenizeAssemblyFunctionDefinition(_AST.operations[i]);
				break;
			case 'AssemblyFor':
				tokenSequence += tokenizeAssemblyFor(_AST.operations[i]);
				break;
			case 'Identifier':
				tokenSequence += tokenizeIdentifier(_AST.operations[i]);
				break;
			case 'AssemblyBlock':
				tokenSequence += tokenizeAssemblyBlock(_AST.operations[i]);
				break;
			case 'HexNumber':
				tokenSequence += tokenizeHexNumber(_AST.operations[i]);
				break;
			default:
				throw "error";
		}
	}
	tokenSequence += (' ' + '}');
	return tokenSequence;
}

function tokenizeMemberAccess(_AST){
	var tokenSequence = '';
	tokenSequence += tokenizeExpression(_AST.expression);
	tokenSequence += (' ' + '.');
	tokenSequence += (' ' + global.identifier_dict[_AST.type][_AST.memberName]);
	return tokenSequence;
}

function tokenizeElementaryTypeNameExpression(_AST){
	var tokenSequence = '';
	if(_AST.typeName != null){
		tokenSequence += tokenizeTypeName(_AST.typeName);
	}
    return tokenSequence;
}

function tokenizeNewExpression(_AST){
	var tokenSequence = '';
	tokenSequence += (' ' + 'new');
	if(_AST.typeName != null){
		tokenSequence += tokenizeTypeName(_AST.typeName);
	}
    return tokenSequence;
}

function tokenizeTupleExpression(_AST){
	var tokenSequence = '';
	tokenSequence += (' ' + '(');
	for(var i = 0; i < _AST.components.length; i++){
		if(i != 0){
			tokenSequence += (' ' + ',');
		}
		tokenSequence += tokenizeExpression(_AST.components[i]);
	}
	tokenSequence += (' ' + ')');
	return tokenSequence;
}

function tokenizeStringLiteral(_AST){
	return (' ' + global.identifier_dict[_AST.type][_AST.value]);
}

function tokenizeBooleanLiteral(_AST){
	return (' ' + global.identifier_dict[_AST.type][_AST.value]);
}

function tokenizeLabelDefinition(_AST){
	return (' ' + global.identifier_dict[_AST.type][_AST.name]);
}

function tokenizeAssemblyAssignment(_AST){
	var tokenSequence = '';
	if(_AST.names.length != 0){
		for(var i = 0; i < _AST.names.length; i++){
			tokenSequence += ' ';
			tokenSequence += tokenizeExpression(_AST.names[i]);
		}
	}
	tokenSequence += (' ' + ':=');
	tokenSequence += tokenizeExpression(_AST.expression);
	return tokenSequence;
}

function tokenizeAssemblyLocalDefinition(_AST){
	var tokenSequence = '';
	tokenSequence += (' ' + 'let');
	if(_AST.names.length != 0){
		for(var i = 0; i < _AST.names.length; i++){
			tokenSequence += ' ';
			tokenSequence += tokenizeExpression(_AST.names[i]);
		}
	}
	if(_AST.expression != null){
		tokenSequence += (' ' + ':=');
		tokenSequence += tokenizeExpression(_AST.expression);
	}
	return tokenSequence;
}

function tokenizeAssemblyCall(_AST){
	var tokenSequence = '';
	tokenSequence += (' ' + global.identifier_dict[_AST.type][_AST.functionName]);
	tokenSequence += (' ' + '(');
	if(_AST.arguments.length != 0){
		tokenSequence += tokenizeArguments(_AST.arguments);
	}
	tokenSequence += (' ' + ')');
	return tokenSequence;
}

function tokenizeAssemblySwitch(_AST){
	var tokenSequence = '';
	tokenSequence += (' ' + 'switch');
	tokenSequence += tokenizeExpression(_AST.expression);
	for(var i = 0; i < _AST.cases.length; i++){
		switch(_AST.cases[i].type){
			case 'AssemblyCase':
				tokenSequence += tokenizeAssemblyCase(_AST.cases[i]);
				break;
			default:
				throw "error";
		}

	}
	return tokenSequence;
}

function tokenizeAssemblyCase(_AST){
	var tokenSequence = '';
	tokenSequence += (' ' + 'case');
	if(_AST.value != null){
		tokenSequence += tokenizeExpression(_AST.value);
	}
	if(_AST.block != null){
		switch(_AST.block.type){
			case 'AssemblyBlock':
				tokenSequence += tokenizeAssemblyBlock(_AST.block);
				break;
			default:
				throw "error";
		}
	}
	return tokenSequence;
}

function tokenizeAssemblyIf(_AST){
	var tokenSequence = '';
	tokenSequence += (' ' + 'if');
	if(_AST.condition != null){
		tokenSequence += tokenizeCondition(_AST.condition);
	}
	if(_AST.body != null){
		switch(_AST.body.type){
			case 'AssemblyBlock':
				tokenSequence += tokenizeAssemblyBlock(_AST.body);
				break;
			default:
				throw "error";
		}
	}
	return tokenSequence;
}

function tokenizeAssemblyFunctionDefinition(_AST){
	var tokenSequence = '';
	tokenSequence += (' ' + 'function' + ' ' + global.identifier_dict[_AST.type][_AST.name]);
	tokenSequence += (' ' + '(');
	if(_AST.arguments.length != 0){
		tokenSequence += tokenizeArguments(_AST.arguments);
	}
	tokenSequence += (' ' + ')');
	if(_AST.returnArguments.length != 0){
    	tokenSequence += (' ' + 'returns' + ' ' + '(');
		tokenSequence += tokenizeParameters(_AST.returnArguments);
		tokenSequence += (' ' + ')');
	}
	if(_AST.body != null){
		switch(_AST.body.type){
			case 'AssemblyBlock':
				tokenSequence += tokenizeAssemblyBlock(_AST.body);
				break;
			default:
				throw "error";
		}
	}
	return tokenSequence;
}
function tokenizeAssemblyFor(_AST){
	var tokenSequence = '';
	tokenSequence += (' ' + 'for');
	if(_AST.pre != null){
		switch(_AST.pre.type){
			case 'AssemblyBlock':
				tokenSequence += tokenizeAssemblyBlock(_AST.pre);
				break;
			default:
				throw "error";
		}
	}
	if(_AST.condition != null){
		tokenSequence += tokenizeCondition(_AST.condition);
	}
	if(_AST.post != null){
		switch(_AST.post.type){
			case 'AssemblyBlock':
				tokenSequence += tokenizeAssemblyBlock(_AST.post);
				break;
			default:
				throw "error";
		}
	}
	if(_AST.body != null){
		switch(_AST.body.type){
			case 'AssemblyBlock':
				tokenSequence += tokenizeAssemblyBlock(_AST.body);
				break;
			default:
				throw "error";
		}
	}
	return tokenSequence;
}



function tokenizeHexNumber(_AST){
	return (' ' + global.identifier_dict[_AST.type][_AST.value]);
}

function tokenizeDecimalNumber(_AST){
	return (' ' + global.identifier_dict[_AST.type][_AST.value]);
}

function tokenizeVariableDeclarationStatement(_AST){
	var tokenSequence = '';
	for(var i =0; i < _AST.variables.length; i++){
		if(_AST.variables[i] != null){
			switch(_AST.variables[i].type){
				case 'VariableDeclaration':
					tokenSequence += tokenizeVariableDeclaration(_AST.variables[i]);
					break;
				default:
					throw "error";
			}
		}
	}
	if(_AST.initialValue != null){
		tokenSequence += (' ' + '=');
		tokenSequence += tokenizeExpression(_AST.initialValue);
	}
	return tokenSequence;
}

function tokenizeInheritanceSpecifier(_AST){
	var tokenSequence = '';
	if(_AST.baseName != null){
		tokenSequence += tokenizeTypeName(_AST.baseName);
	}
	if(_AST.arguments.length != 0){
		tokenSequence += tokenizeArguments(_AST.arguments);
	}
	return tokenSequence;
}

function tokenizeUserDefinedTypeName(_AST){
	return (' ' + global.identifier_dict[_AST.type][_AST.namePath]);
}

function tokenizeModifierInvocation(_AST){
	var tokenSequence = '';
	tokenSequence += (' ' + global.identifier_dict[_AST.type][_AST.name]);
	if(_AST.arguments != null){
		tokenSequence += (' ' + '(');
		tokenSequence += tokenizeArguments(_AST.arguments);
		tokenSequence += (' ' + ')');
	}
	return tokenSequence;
}

function tokenizeMapping(_AST){
	var tokenSequence = '';
	tokenSequence += (' ' + 'mapping' + ' ' + '(');
	if(_AST.keyType != null){
		tokenSequence += tokenizeTypeName(_AST.keyType);
	}
	tokenSequence += (' ' + '=>');
	if(_AST.valueType != null){
		tokenSequence += tokenizeTypeName(_AST.valueType);
	}
	tokenSequence += (' ' + ')');
	return tokenSequence;
}

function tokenizeTypeName(_AST){
	switch(_AST.type){
		case 'ArrayTypeName':
			return tokenizeArrayTypeName(_AST);
		case 'ElementaryTypeName':
			return tokenizeElementaryTypeName(_AST);
		case 'UserDefinedTypeName':
			return tokenizeUserDefinedTypeName(_AST);
		case 'Mapping':
			return tokenizeMapping(_AST);
		case 'FunctionTypeName':
			return tokenizeFunctionTypeName(_AST);
		case 'ImportDirective':
			return tokenizeImportDirective(_AST);
		default:
			throw "error";
        	return '';
	}
}

function tokenizeHexLiteral(_AST){
	return (' ' + global.identifier_dict[_AST.type][_AST.value]);
}

function tokenizeEnumValue(_AST){
	return (' ' + global.identifier_dict[_AST.type][_AST.name]);
}

function tokenizeConditional(_AST){
	var tokenSequence = '';
	if(_AST.condition != null){
		tokenSequence += tokenizeExpression(_AST.condition);
	}
	tokenSequence += (' ' + '?');
	if(_AST.trueExpression != null){
		tokenSequence += tokenizeExpression(_AST.trueExpression);
	}
	tokenSequence += (' ' + ':');
	if(_AST.falseExpression != null){
		tokenSequence += tokenizeExpression(_AST.falseExpression);
	}
    return tokenSequence;
}

function tokenizeDoWhileStatement(_AST){
	var tokenSequence = '';
	tokenSequence += (' ' + 'do');
	if(_AST.body.type == 'Block'){
		tokenSequence += tokenizeBlock(_AST.body);
	}else{
		tokenSequence += tokenizeStatement(_AST.body);
	}
	tokenSequence += (' ' + 'while' + ' ' + '(');
	if(_AST.condition != null){
		tokenSequence += tokenizeCondition(_AST.condition);
	}
	tokenSequence += (' ' + ')');
	return tokenSequence;
}

function tokenizeImportDirective(_AST){
	var tokenSequence = '';
	tokenSequence += (' ' + 'import');
	if(_AST.unitAlias != null){
		throw "error";
	}
	if(_AST.symbolAliases != null){
		tokenSequence += (' ' + '{');
		for(var i = 0; i < _AST.symbolAliases.length; i++){
			tokenSequence += _AST.symbolAliases[i][0];
		}
		tokenSequence += (' ' + '}' + ' ' + 'from');
	}
	tokenSequence += (' ' + '"' + ' ' + global.identifier_dict[_AST.type][_AST.path] + ' ' + '"');
	return tokenSequence;
}

function tokenizeFunctionTypeName(_AST){
	var tokenSequence = '';
	tokenSequence += (' ' + 'function' + ' ' + '(');
	if(_AST.parameterTypes.length != 0){
		tokenSequence += tokenizeParameters(_AST.parameterTypes);
	}
	tokenSequence += (' ' + ')');
	if(_AST.stateMutability != null){
		tokenSequence += (' ' + _AST.stateMutability);
	}
	if(_AST.visibility != null){
		tokenSequence += (' ' + _AST.visibility);
	}
	if(_AST.returnTypes.length != 0){
		tokenSequence += (' ' + 'returns' + ' ' + '(');
		tokenSequence += (' ' + _AST.visibility);
		tokenSequence += (' ' + ')');
	}
	return tokenSequence;
}
