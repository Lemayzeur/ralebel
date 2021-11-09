from strings_with_arrows import *

import string 

''' CONSTANTS'''

DIGITS = '0123456789'
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS

TT_INT		= 'INT'
TT_FLOAT	= 'FLOAT'
TT_IDENTIFIER = 'IDENTIFIER'
TT_KEYWORD 	= 'KEYWORD'
TT_PLUS		= 'PLUS'
TT_MINUS	= 'MIN'
TT_MUL		= 'MUL'
TT_DIV		= 'DIV'
TT_POW		= 'POW'
TT_EQ		= 'EQ'
TT_LPAREN	= 'LPAREN'
TT_RPAREN	= 'RPAREN'
TT_EE 		= 'EE'
TT_NE 		= 'NE'
TT_LT 		= 'LT'
TT_GT 		= 'GT'
TT_LTE		= 'LTE'
TT_GTE 		= 'GTE'
TT_EOF		= 'EOF'

KEYWORDS = [
	'VAR', 'var',
	'AK', 'ak',
	'OU', 'ou',
	'NEG', 'neg'
]

''' ERRORS'''
class Error:
	def __init__(self, pos_start, pos_end, error_name, details):
		self.pos_start = pos_start 
		self.pos_end = pos_end 
		self.error_name = error_name 
		self.details = details

	def as_string(self):
		result = f'{self.error_name}: {self.details}'
		result += f'\nFichye <{self.pos_start.filename}>, liy {self.pos_start.line + 1}'
		return result


class IllegalCharError(Error):
	def __init__(self, pos_start, pos_end, details):
		super().__init__(pos_start, pos_end, 'Karaktè Envalid', details)

class ExpectedCharError(Error):
	def __init__(self, pos_start, pos_end, details):
		super().__init__(pos_start, pos_end, 'Nou atann karaktè', details)

class InvalidSyntaxError(Error):
	def __init__(self, pos_start, pos_end, details=''):
		super().__init__(pos_start, pos_end, 'Sentaks Envalid', details)

class RTError(Error):
	def __init__(self, pos_start, pos_end, details, context):
		super().__init__(pos_start, pos_end, 'Erè Lansman', details)
		self.context = context

	def as_string(self):
		result  = self.generate_traceback()
		result += f'{self.error_name}: {self.details}'
		result += '\n\n' + string_with_arrows(self.pos_start.filetext, self.pos_start, self.pos_end)
		return result

	def generate_traceback(self):
		result = ''
		pos = self.pos_start
		ctx = self.context

		while ctx:
			result = f'  Fichye <{pos.filename}>, liy {str(pos.line + 1)}, nan {ctx.display_name}\n' + result
			pos = ctx.parent_entry_pos
			ctx = ctx.parent

		return 'Retrase (dènye ekzekisyon an):\n' + result

''' POSITION'''
class Position:
	def __init__(self, index, line, col, filename, filetext):
		self.index = index
		self.line = line 
		self.col = col 
		self.filename = filename 
		self.filetext = filetext

	def advance(self, current_char=None):
		self.index += 1
		self.col += 1

		if current_char == '\n':
			self.line += 1
			self.col = 0
		return self  

	def copy(self):
		return Position(self.index, self.line, self.col, self.filename, self.filetext)


''' TOKENS'''
class Token:
	def __init__(self, type_, value=None, pos_start=None, pos_end=None):
		self.type = type_
		self.value = value

		if pos_start:
			self.pos_start = pos_start.copy()
			self.pos_end = pos_start.copy()
			self.pos_end.advance()

		if pos_end:
			self.pos_start = pos_start
			self.pos_end = pos_end

	def matches(self, type_, value):
		return self.type == type_ and self.value == value

	def __repr__(self):
		if self.value:
			return '%s:%s' % (self.type, self.value)
		return self.type


''' LEXER'''
class Lexer:
	def __init__(self, filename, text):
		self.filename = filename 
		self.text = text 
		self.pos = Position(-1, 0, -1, self.filename, text)
		self.current_char = None
		self.advance()

	@property
	def clean_text(self):
		return self.text.strip().replace('\t','').replace('\n','').replace(' ', '')

	def advance(self):
		self.pos.advance(self.current_char)
		# print(self.pos, self.text)
		self.current_char = self.text[self.pos.index] if self.pos.index < len(self.text) else None

	def make_tokens(self):
		tokens = []

		# print(self.current_char)
		while self.current_char != None:
			if self.current_char in ' \t':
				self.advance()
			elif self.current_char in DIGITS:
				tokens.append(self.make_number())
			elif self.current_char in LETTERS:
				tokens.append(self.make_identifier())
				# self.advance()
			elif self.current_char  == '+':
				tokens.append(Token(TT_PLUS, pos_start=self.pos))
				self.advance()
			elif self.current_char  == '-':
				tokens.append(Token(TT_MINUS, pos_start=self.pos))
				self.advance()
			elif self.current_char  == '*':
				tokens.append(Token(TT_MUL, pos_start=self.pos))
				self.advance()
			elif self.current_char  == '/':
				tokens.append(Token(TT_DIV, pos_start=self.pos))
				self.advance()
			elif self.current_char  == '^':
				tokens.append(Token(TT_POW, pos_start=self.pos))
				self.advance()
			elif self.current_char  == '(':
				tokens.append(Token(TT_LPAREN, pos_start=self.pos))
				self.advance()
			elif self.current_char  == ')':
				tokens.append(Token(TT_RPAREN, pos_start=self.pos))
				self.advance()
			elif self.current_char == '!':
				token, error = self.make_not_equals()
				if error: return [], error 
				tokens.append(token)
			elif self.current_char == '=':
				tokens.append(self.make_equals())
			elif self.current_char == '<':
				tokens.append(self.make_less_than())
			elif self.current_char == '>':
				tokens.append(self.make_greater_than())
			else:
				pos_start = self.pos.copy()
				char = self.current_char 
				self.advance()
				return [], IllegalCharError(pos_start, self.pos, f"'{char}'") 

		tokens.append(Token(TT_EOF, pos_start=self.pos))
		return tokens, None

	def make_number(self):
		num_str = '' 
		dot_count = 0
		pos_start = self.pos.copy()

		while self.current_char != None and self.current_char in DIGITS + '.':
			if self.current_char == '.':
				if dot_count == 1:
					break
				dot_count += 1 
				num_str += '.'
			else:
				num_str += self.current_char 
			self.advance()

		if dot_count == 0:
			return Token(TT_INT, int(num_str), pos_start, self.pos)
		return Token(TT_FLOAT, float(num_str), pos_start, self.pos)

	def make_identifier(self):
		id_str = ''
		pos_start = self.pos.copy()

		while self.current_char != None and self.current_char in LETTERS_DIGITS + '_':
			id_str += self.current_char
			self.advance()

		tok_type = TT_KEYWORD if id_str in KEYWORDS else TT_IDENTIFIER
		return Token(tok_type, id_str, pos_start, self.pos)

	def make_not_equals(self):
		pos_start = self.pos.copy()
		self.advance()

		if self.current_char == '=':
			self.advance()
			return Token(TT_NE, pos_start=pos_start, pos_end=self.pos), None

		self.advance()
		return None, ExpectedCharError(pos_start, self.pos, "'=' (aprè siy '!')")

	def make_equals(self):
		tok_type = TT_EQ
		pos_start = self.pos.copy()
		self.advance()

		if self.current_char == '=':
			self.advance()
			tok_type = TT_EE

		return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

	def make_less_than(self):
		tok_type = TT_LT
		pos_start = self.pos.copy()
		self.advance()

		if self.current_char == '=':
			self.advance()
			tok_type = TT_LTE

		return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

	def make_greater_than(self):
		tok_type = TT_GT
		pos_start = self.pos.copy()
		self.advance()

		if self.current_char == '=':
			self.advance()
			tok_type = TT_GTE

		return Token(tok_type, pos_start=pos_start, pos_end=self.pos)


	def __repr__(self):
		if self.value:
			return '%s:%s' % (self.type, self.value)
		return self.type

#######################################
# NODES
#######################################

class NumberNode:
	def __init__(self, tok):
		self.tok = tok

		self.pos_start = self.tok.pos_start
		self.pos_end = self.tok.pos_end

	def __repr__(self):
		return f'{self.tok}'

class VarAccessNode:
	def __init__(self, var_name_tok):
		self.var_name_tok = var_name_tok

		self.pos_start = self.var_name_tok.pos_start
		self.pos_end = self.var_name_tok.pos_end

class VarAssignNode:
	def __init__(self, var_name_tok, value_node, var_access=False):
		self.var_name_tok = var_name_tok
		self.value_node = value_node
		self.var_access = var_access

		self.pos_start = self.var_name_tok.pos_start
		self.pos_end = self.value_node.pos_end

class BinOpNode:
	def __init__(self, left_node, op_tok, right_node):
		self.left_node = left_node
		self.op_tok = op_tok
		self.right_node = right_node

		self.pos_start = self.left_node.pos_start
		self.pos_end = self.right_node.pos_end

	def __repr__(self):
		return f'({self.left_node}, {self.op_tok}, {self.right_node})'

class UnaryOpNode:
	def __init__(self, op_tok, node):
		self.op_tok = op_tok
		self.node = node

		self.pos_start = self.op_tok.pos_start
		self.pos_end = node.pos_end

	def __repr__(self):
		return f'({self.op_tok}, {self.node})'

#######################################
# PARSE RESULT
#######################################

class ParseResult:
	def __init__(self):
		self.error = None
		self.node = None
		self.advance_count = 0

	def register_advancement(self):
		self.advance_count += 1

	def register(self, res):
		self.advance_count += res.advance_count
		if res.error: self.error = res.error
		return res.node


	def success(self, node):
		self.node = node
		return self

	def failure(self, error):
		if not self.error or self.advance_count == 0:
			self.error = error
		return self


#######################################
# PARSER
#######################################

class Parser:
	def __init__(self, tokens):
		self.tokens = tokens
		self.tok_idx = -1
		self.next_tok = None
		self.advance()

	def advance(self, ):
		self.tok_idx += 1
		if self.tok_idx < len(self.tokens):
			self.current_tok = self.tokens[self.tok_idx]
		if (self.tok_idx + 1) < len(self.tokens):
			self.next_tok = self.tokens[self.tok_idx + 1]
		return self.current_tok

	def parse(self):
		res = self.expr()
		if not res.error and self.current_tok.type != TT_EOF:
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				"Nou atann '+', '-', '*', '/', '^', '==', '!=', '<', '>', <=', '>=', 'AK' oubyen 'OU'"
			))
		return res

	###################################

	def atom(self):
		res = ParseResult()
		tok = self.current_tok

		if tok.type in (TT_INT, TT_FLOAT):
			res.register_advancement()
			self.advance()
			return res.success(NumberNode(tok))

		elif tok.type == TT_IDENTIFIER:
			res.register_advancement()
			self.advance()
			return res.success(VarAccessNode(tok))

		elif tok.type == TT_LPAREN:
			res.register_advancement()
			self.advance()
			expr = res.register(self.expr())
			if res.error: return res
			if self.current_tok.type == TT_RPAREN:
				res.register_advancement()
				self.advance()
				return res.success(expr)
			else:
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					"Nou atann ')'"
				))

		return res.failure(InvalidSyntaxError(
			tok.pos_start, tok.pos_end,
			"Nou atann antye | desimal, varyab, '+', '-', '('"
		))

	def power(self):
		return self.bin_op(self.atom, (TT_POW, ), self.factor)

	def factor(self):
		res = ParseResult()
		tok = self.current_tok

		if tok.type in (TT_PLUS, TT_MINUS):
			res.register_advancement()
			self.advance()
			factor = res.register(self.factor())
			if res.error: return res
			return res.success(UnaryOpNode(tok, factor))
		
		return self.power()

	def term(self):
		return self.bin_op(self.factor, (TT_MUL, TT_DIV))

	def arith_expr(self):
		return self.bin_op(self.term, (TT_PLUS, TT_MINUS))

	def comp_expr(self):
		res = ParseResult()

		if self.current_tok.matches(TT_KEYWORD, 'NEG') or self.current_tok.matches(TT_KEYWORD, 'neg'):
			op_tok = self.current_tok
			res.register_advancement()
			self.advance()

			node = res.register(self.comp_expr())
			if res.error: return res
			return res.success(UnaryOpNode(op_tok, node))
		
		node = res.register(self.bin_op(self.arith_expr, (TT_EE, TT_NE, TT_LT, TT_GT, TT_LTE, TT_GTE)))
		
		if res.error:
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				"Expected int, float, identifier, '+', '-', '(' oubyen 'NEG'"
			))

		return res.success(node)


	def expr(self):
		res = ParseResult()

		if self.current_tok.matches(TT_KEYWORD, 'VAR') or self.current_tok.matches(TT_KEYWORD, 'var'):
			res.register_advancement()
			self.advance()

			if self.current_tok.type != TT_IDENTIFIER:
				if self.current_tok.value in KEYWORDS:
					return res.failure(InvalidSyntaxError(
						self.current_tok.pos_start, self.current_tok.pos_end,
						"Itilizasyon mo kle pou non varyab"
					))
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					"Atann yon non varyab"
				))

			var_name = self.current_tok
			res.register_advancement()
			self.advance()

			if self.current_tok.type != TT_EQ:
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					"Atann siy '='"
				))

			res.register_advancement()
			self.advance()
			expr = res.register(self.expr())
			if res.error: return res
			return res.success(VarAssignNode(var_name, expr))

		# Access previous defined variable
		if self.current_tok.type == TT_IDENTIFIER and (self.next_tok is not None and self.next_tok.type == TT_EQ):
			var_name = self.current_tok
			res.register_advancement()
			self.advance()
			res.register_advancement()
			self.advance()
			expr = res.register(self.expr())
			if res.error: return res
			return res.success(VarAssignNode(var_name, expr, var_access=True))

		node = res.register(
			self.bin_op(self.comp_expr, ((TT_KEYWORD, 'AK'), (TT_KEYWORD, 'ak'), (TT_KEYWORD, 'OU'), (TT_KEYWORD, 'ou')) )
		)

		if res.error:
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				"Atann 'VAR', antye, desimal, non varyav, '+', '-' oubyen '('"
			))

		return res.success(node)

	###################################

	def bin_op(self, func_a, ops, func_b=None):
		if func_b == None:
			func_b = func_a
		
		res = ParseResult()
		left = res.register(func_a())
		if res.error: return res

		while self.current_tok.type in ops or (self.current_tok.type, self.current_tok.value) in ops:
			op_tok = self.current_tok
			res.register_advancement()
			self.advance()
			right = res.register(func_b())
			if res.error: return res
			left = BinOpNode(left, op_tok, right)

		return res.success(left)

#######################################
# RUNTIME RESULT
#######################################

class RTResult:
	def __init__(self):
		self.value = None
		self.error = None

	def register(self, res):
		if res.error: self.error = res.error
		return res.value

	def success(self, value):
		self.value = value
		return self

	def failure(self, error):
		self.error = error
		return self

''' VALUES'''
class Number:
	def __init__(self, value):
		self.value = value
		self.set_pos()
		self.set_context()

	def set_pos(self, pos_start=None, pos_end=None):
		self.pos_start = pos_start
		self.pos_end = pos_end
		return self

	def set_context(self, context=None):
		self.context = context
		return self

	def added_to(self, other):
		if isinstance(other, Number):
			return Number(self.value + other.value).set_context(self.context), None

	def subbed_by(self, other):
		if isinstance(other, Number):
			return Number(self.value - other.value).set_context(self.context), None

	def multed_by(self, other):
		if isinstance(other, Number):
			return Number(self.value * other.value).set_context(self.context), None

	def dived_by(self, other):
		if isinstance(other, Number):
			if other.value == 0:
				return None, RTError(
					other.pos_start, other.pos_end,
					'Divize pa zewo',
					self.context
				)

			return Number(self.value / other.value).set_context(self.context), None

	def powed_by(self, other):
		if isinstance(other, Number):
			if isinstance(other.value, float):
				return None, RTError(
					other.pos_start, other.pos_end,
					'Ekspozisyon ak desimal',
					self.context
				)
			return Number(self.value ** other.value).set_context(self.context), None

	def get_comparison_eq(self, other):
		if isinstance(other, Number):
			return Number(int(self.value == other.value)).set_context(self.context), None

	def get_comparison_ne(self, other):
		if isinstance(other, Number):
			return Number(int(self.value != other.value)).set_context(self.context), None

	def get_comparison_lt(self, other):
		if isinstance(other, Number):
			return Number(int(self.value < other.value)).set_context(self.context), None

	def get_comparison_gt(self, other):
		if isinstance(other, Number):
			return Number(int(self.value > other.value)).set_context(self.context), None

	def get_comparison_lte(self, other):
		if isinstance(other, Number):
			return Number(int(self.value <= other.value)).set_context(self.context), None

	def get_comparison_gte(self, other):
		if isinstance(other, Number):
			return Number(int(self.value >= other.value)).set_context(self.context), None

	def anded_by(self, other):
		if isinstance(other, Number):
			return Number(int(self.value and other.value)).set_context(self.context), None

	def ored_by(self, other):
		if isinstance(other, Number):
			return Number(int(self.value or other.value)).set_context(self.context), None

	def notted(self):
		return Number(1 if self.value == 0 else 0).set_context(self.context), None

	def copy(self):
		copy = Number(self.value)
		copy.set_pos(self.pos_start, self.pos_end)
		copy.set_context(self.context)
		return copy

	def __repr__(self):
		return str(self.value)

#######################################
# CONTEXT
#######################################

class Context:
	def __init__(self, display_name, parent=None, parent_entry_pos=None):
		self.display_name = display_name
		self.parent = parent
		self.parent_entry_pos = parent_entry_pos
		self.symbol_table = None

''' SYMBOL TABLE '''
class SymbolTable:
	def __init__(self):
		self.symbols = {}
		self.parent = None

	def get(self, name):
		value = self.symbols.get(name, None)
		if value == None and self.parent:
			return self.parent.get(name)
		return value

	def contains(self, name):
		exists = name in self.symbols
		if not exists and self.parent:
			return name in self.parent
		return exists

	def set(self, name, value):
		self.symbols[name] = value

	def remove(self, name):
		del self.symbols[name]

''' INTERPRETER'''

class Interpreter:
	def visit(self, node, context):
		method_name = f'visit_{type(node).__name__}'
		method = getattr(self, method_name, self.no_visit_method)
		return method(node, context)

	def no_visit_method(self, node, context):
		raise Exception(f'No visit_{type(node).__name__} method defined')

	###################################

	def visit_NumberNode(self, node, context):
		return RTResult().success(
			Number(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
		)

	def visit_VarAccessNode(self, node, context):
		res = RTResult()
		var_name = node.var_name_tok.value
		value = context.symbol_table.get(var_name)

		if not value:
			return res.failure(RTError(
				node.pos_start, node.pos_end,
				f"'{var_name}' pa defini",
				context
			))

		value = value.copy().set_pos(node.pos_start, node.pos_end)
		return res.success(value)

	def visit_VarAssignNode(self, node, context):
		var_access = getattr(node, 'var_access', False)
		res = RTResult()
		var_name = node.var_name_tok.value
		
		if var_access:
			if not context.symbol_table.contains(var_name):
				return res.failure(RTError(
					node.pos_start, node.pos_end,
					f"'{var_name}' pa te defini",
					context
				))

		value = res.register(self.visit(node.value_node, context))
		if res.error: return res

		context.symbol_table.set(var_name, value)
		return res.success(value)

	def visit_BinOpNode(self, node, context):
		res = RTResult()
		left = res.register(self.visit(node.left_node, context))
		if res.error: return res
		right = res.register(self.visit(node.right_node, context))
		if res.error: return res

		if node.op_tok.type == TT_PLUS:
			result, error = left.added_to(right)
		elif node.op_tok.type == TT_MINUS:
			result, error = left.subbed_by(right)
		elif node.op_tok.type == TT_MUL:
			result, error = left.multed_by(right)
		elif node.op_tok.type == TT_DIV:
			result, error = left.dived_by(right)
		elif node.op_tok.type == TT_POW:
			result, error = left.powed_by(right)
		elif node.op_tok.type == TT_EE:
			result, error = left.get_comparison_eq(right)
		elif node.op_tok.type == TT_NE:
			result, error = left.get_comparison_ne(right)
		elif node.op_tok.type == TT_LT:
			result, error = left.get_comparison_lt(right)
		elif node.op_tok.type == TT_GT:
			result, error = left.get_comparison_gt(right)
		elif node.op_tok.type == TT_LTE:
			result, error = left.get_comparison_lte(right)
		elif node.op_tok.type == TT_GTE:
			result, error = left.get_comparison_gte(right)
		elif node.op_tok.matches(TT_KEYWORD, 'AK') or node.op_tok.matches(TT_KEYWORD, 'ak'):
			result, error = left.anded_by(right)
		elif node.op_tok.matches(TT_KEYWORD, 'OU') or node.op_tok.matches(TT_KEYWORD, 'ou'):
			result, error = left.ored_by(right)

		if error:
			return res.failure(error)
		else:
			return res.success(result.set_pos(node.pos_start, node.pos_end))

	def visit_UnaryOpNode(self, node, context):
		res = RTResult()
		number = res.register(self.visit(node.node, context))
		if res.error: return res

		error = None

		if node.op_tok.type == TT_MINUS:
			number, error = number.multed_by(Number(-1))
		elif node.op_tok.matches(TT_KEYWORD, 'NEG') or node.op_tok.matches(TT_KEYWORD, 'neg'):
			number, error = number.notted()

		if error:
			return res.failure(error)
		else:
			return res.success(number.set_pos(node.pos_start, node.pos_end))

global_symbol_table = SymbolTable()
global_symbol_table.set("ANYEN", Number(0))
global_symbol_table.set("anyen", Number(0))
global_symbol_table.set("FO", Number(0))
global_symbol_table.set("fo", Number(0))
global_symbol_table.set("VRE", Number(1))
global_symbol_table.set("vre", Number(1))

def run(fn, text):
	# Generate tokens
	lexer = Lexer(fn, text)
	if lexer.clean_text != '':
		tokens, error = lexer.make_tokens()
		if error: return None, error
		
		# Generate AST
		parser = Parser(tokens)
		ast = parser.parse()
		if ast.error: return None, ast.error

		# Run program
		interpreter = Interpreter()
		context = Context('<program>')
		context.symbol_table = global_symbol_table
		result = interpreter.visit(ast.node, context)

		return result.value, result.error
	return '', None