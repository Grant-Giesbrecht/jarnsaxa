from jarnsaxa import wrap_text, ensureWhitespace, parse_idx, StringIdx

def test_wrap_text():
	
	wt1 = wrap_text("test string", width=6)
	wt2 = wrap_text("test string")
	assert wt1 == "test\nstring"
	assert wt2 == "test string"

def test_ensure_whitespace():
	
	ew1 = ensureWhitespace("test [1,2,3]+\t19", ",[]+-*/")
	assert ew1 == "test [ 1 , 2 , 3 ] +\t19"
	
	ew1 = ensureWhitespace("+", ",[]+-*/")
	assert ew1 == "+"

def test_parse_idx():
	
	w1 = parse_idx("This is a test")
	assert len(w1) == 4
	assert w1[0].str == "This"
	assert w1[0].idx == 0
	assert w1[0].idx_end == 4
	assert w1[1].str == "is"
	assert w1[1].idx == 5
	assert w1[1].idx_end == 7
	assert w1[2].str == "a"
	assert w1[2].idx == 8
	assert w1[2].idx_end == 9
	assert w1[3].str == "test"
	assert w1[3].idx == 10
	assert w1[3].idx_end == 14


def test_stringidx_str_repr():
	
	si = StringIdx("Test", 0, 4)
	assert si.__str__() == "[0]\"Test\""
	assert si.__repr__() == "[0]\"Test\""
	