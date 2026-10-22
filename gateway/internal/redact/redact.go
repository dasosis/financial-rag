// Package redact replaces personal data in prompts with typed placeholders
// ("[EMAIL]") before they are cached or sent to a provider.
//
// High precision over recall: every detector needs a value's own format, and
// where one exists, its checksum. A false positive here would change what the
// model reads, and in a 10-K most of what looks like numbers is figures, years
// and page references; a test runs the redactor over both sample filings and
// every benchmark question and fails if any of those changes.
//
//	EMAIL    name@domain.tld
//	PHONE    formatted only: (425) 882-8080, 425-882-8080, 425.882.8080,
//	         1-800-555-0100, +44 20 7946 0958, +91 98765 43210 (spaces
//	         only after a +country code: a table row is digits and spaces)
//	CARD     13-19 digits, plain or in 4-4-4-4 / 4-6-5 groups, passing Luhn
//	SSN      dashed only: 123-45-6789, with a valid area, group and serial
//	PAN      India's Permanent Account Number: ABCPE1234F
//	AADHAAR  12 digits, plain or 4-4-4, first 2-9, passing Verhoeff
//
// A group of digits is only taken whole: never a slice of a longer run of
// digits (a row of years "2024 2023 2022 2021" is not a card number), and a
// run whose every group is a year is never a card or Aadhaar number.
package redact

import (
	"regexp"
	"strings"
)

// Kind is a type of personal data, and its placeholder's name.
type Kind string

const (
	Email   Kind = "EMAIL"
	Phone   Kind = "PHONE"
	Card    Kind = "CARD"
	SSN     Kind = "SSN"
	PAN     Kind = "PAN"
	Aadhaar Kind = "AADHAAR"
)

// Kinds are all the detectors, in the order they run: the longer digit
// formats first, so a card number is never half-read as a phone number.
var Kinds = []Kind{Email, Card, Aadhaar, SSN, Phone, PAN}

type detector struct {
	kind  Kind
	re    *regexp.Regexp
	valid func(match string) bool
	// digitsOnly: the value is a run of digits and separators, which must not
	// be part of a longer run
	digitsOnly bool
}

var detectors = map[Kind]detector{
	Email: {kind: Email, re: regexp.MustCompile(
		`[A-Za-z0-9._%+\-]+@[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?)*\.[A-Za-z]{2,24}`)},
	Card: {kind: Card, digitsOnly: true, valid: validCard, re: regexp.MustCompile(
		`\d{4} \d{4} \d{4} \d{4}(?: \d{1,3})?|\d{4}-\d{4}-\d{4}-\d{4}(?:-\d{1,3})?|\d{4} \d{6} \d{4,5}|\d{4}-\d{6}-\d{4,5}|\d{13,19}`)},
	Aadhaar: {kind: Aadhaar, digitsOnly: true, valid: validAadhaar, re: regexp.MustCompile(
		`[2-9]\d{3} \d{4} \d{4}|[2-9]\d{3}-\d{4}-\d{4}|[2-9]\d{11}`)},
	SSN: {kind: SSN, digitsOnly: true, valid: validSSN, re: regexp.MustCompile(`\d{3}-\d{2}-\d{4}`)},
	Phone: {kind: Phone, digitsOnly: true, valid: validPhone, re: regexp.MustCompile(
		`\+\d{1,3}(?:[ .\-]?\(\d{1,4}\))?(?:[ .\-]\d{1,5}){2,5}` +
			`|1[.\-]\d{3}[.\-]\d{3}[.\-]\d{4}` +
			`|\(\d{3}\) ?\d{3}[.\-]\d{4}` +
			`|\d{3}-\d{3}-\d{4}|\d{3}\.\d{3}\.\d{4}`)},
	PAN: {kind: PAN, valid: validPAN, re: regexp.MustCompile(`[A-Z]{3}[ABCFGHJLPT][A-Z]\d{4}[A-Z]`)},
}

// Redactor redacts the kinds it is given.
type Redactor struct {
	detectors []detector
}

// New returns a redactor for these kinds (all of them if none are given).
func New(kinds ...Kind) *Redactor {
	if len(kinds) == 0 {
		kinds = Kinds
	}
	want := map[Kind]bool{}
	for _, k := range kinds {
		want[k] = true
	}
	r := &Redactor{}
	for _, k := range Kinds {
		if want[k] {
			r.detectors = append(r.detectors, detectors[k])
		}
	}
	return r
}

// Text returns s with each piece of personal data replaced by its placeholder,
// and how many of each kind were replaced. It never reports the values.
func (r *Redactor) Text(s string) (string, map[Kind]int) {
	var counts map[Kind]int
	for _, d := range r.detectors {
		found := d.find(s)
		if len(found) == 0 {
			continue
		}
		var out strings.Builder
		last := 0
		for _, loc := range found {
			out.WriteString(s[last:loc[0]])
			out.WriteString("[" + string(d.kind) + "]")
			last = loc[1]
		}
		out.WriteString(s[last:])
		s = out.String()
		if counts == nil {
			counts = map[Kind]int{}
		}
		counts[d.kind] += len(found)
	}
	return s, counts
}

// Match is a piece of personal data found in a text.
type Match struct {
	Kind       Kind
	Start, End int // in the text as it was when this kind was looked for
	Value      string
}

// Find lists what Text would replace, kind by kind (for tests and tuning:
// the gateway itself never looks at the values).
func (r *Redactor) Find(s string) []Match {
	var out []Match
	for _, d := range r.detectors {
		found := d.find(s)
		for _, loc := range found {
			out = append(out, Match{Kind: d.kind, Start: loc[0], End: loc[1], Value: s[loc[0]:loc[1]]})
		}
		if len(found) > 0 {
			s, _ = (&Redactor{detectors: []detector{d}}).Text(s)
		}
	}
	return out
}

func (d detector) find(s string) [][2]int {
	if !d.re.MatchString(s) {
		return nil
	}
	var out [][2]int
	for _, loc := range d.re.FindAllStringIndex(s, -1) {
		start, end := loc[0], loc[1]
		if wordBounded(s, start, end, d.digitsOnly) && (d.valid == nil || d.valid(s[start:end])) {
			out = append(out, [2]int{start, end})
		}
	}
	return out
}

func isDigit(b byte) bool { return b >= '0' && b <= '9' }
func isAlnum(b byte) bool {
	return isDigit(b) || (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')
}

// wordBounded: the match isn't glued to letters or digits around it, and a
// digit format isn't a slice of a longer run of digits ("1 2024 2023 …",
// "0000950170-24-087843", "245,122.45").
func wordBounded(s string, start, end int, digits bool) bool {
	if start > 0 {
		b := s[start-1]
		if isAlnum(b) || b == '_' || b == '@' {
			return false
		}
		if digits && start > 1 && strings.IndexByte(" -.,/", b) >= 0 && isDigit(s[start-2]) {
			return false
		}
		if digits && b == '$' {
			return false // an amount
		}
	}
	if end < len(s) {
		b := s[end]
		if isAlnum(b) || b == '_' || b == '@' {
			return false
		}
		if digits && end+1 < len(s) && strings.IndexByte(" -.,/%", b) >= 0 && isDigit(s[end+1]) {
			return false
		}
		if digits && b == '%' {
			return false
		}
	}
	return true
}

func digitsOf(s string) []int {
	var out []int
	for i := 0; i < len(s); i++ {
		if isDigit(s[i]) {
			out = append(out, int(s[i]-'0'))
		}
	}
	return out
}

// allYears: every group of digits is a plausible year, 1900-2099.
func allYears(s string) bool {
	groups := strings.FieldsFunc(s, func(r rune) bool { return r == ' ' || r == '-' })
	if len(groups) < 2 {
		return false
	}
	for _, g := range groups {
		if len(g) != 4 || !(strings.HasPrefix(g, "19") || strings.HasPrefix(g, "20")) {
			return false
		}
	}
	return true
}

func validCard(s string) bool {
	d := digitsOf(s)
	if len(d) < 13 || len(d) > 19 || allYears(s) {
		return false
	}
	return luhn(d)
}

func luhn(d []int) bool {
	sum := 0
	for i := len(d) - 1; i >= 0; i-- {
		v := d[i]
		if (len(d)-1-i)%2 == 1 {
			v *= 2
			if v > 9 {
				v -= 9
			}
		}
		sum += v
	}
	return sum%10 == 0
}

func validAadhaar(s string) bool {
	d := digitsOf(s)
	return len(d) == 12 && !allYears(s) && verhoeff(d)
}

// The Verhoeff check: dihedral-group multiplication and a permutation table.
var (
	verhoeffD = [10][10]int{
		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 2, 3, 4, 0, 6, 7, 8, 9, 5},
		{2, 3, 4, 0, 1, 7, 8, 9, 5, 6}, {3, 4, 0, 1, 2, 8, 9, 5, 6, 7},
		{4, 0, 1, 2, 3, 9, 5, 6, 7, 8}, {5, 9, 8, 7, 6, 0, 4, 3, 2, 1},
		{6, 5, 9, 8, 7, 1, 0, 4, 3, 2}, {7, 6, 5, 9, 8, 2, 1, 0, 4, 3},
		{8, 7, 6, 5, 9, 3, 2, 1, 0, 4}, {9, 8, 7, 6, 5, 4, 3, 2, 1, 0},
	}
	verhoeffP = [8][10]int{
		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 5, 7, 6, 2, 8, 3, 0, 9, 4},
		{5, 8, 0, 3, 7, 9, 6, 1, 4, 2}, {8, 9, 1, 6, 0, 4, 3, 5, 2, 7},
		{9, 4, 5, 3, 1, 2, 6, 8, 7, 0}, {4, 2, 8, 6, 5, 7, 3, 9, 0, 1},
		{2, 7, 9, 3, 8, 0, 6, 4, 1, 5}, {7, 0, 4, 6, 9, 1, 3, 2, 5, 8},
	}
)

func verhoeff(d []int) bool {
	c := 0
	for i := range d {
		c = verhoeffD[c][verhoeffP[i%8][d[len(d)-1-i]]]
	}
	return c == 0
}

func validSSN(s string) bool {
	area, group, serial := s[0:3], s[4:6], s[7:11]
	if area == "000" || area == "666" || area[0] == '9' {
		return false
	}
	return group != "00" && serial != "0000"
}

func validPhone(s string) bool {
	n := len(digitsOf(s))
	if strings.HasPrefix(s, "+") {
		return n >= 8 && n <= 15
	}
	return n == 10 || n == 11
}

func validPAN(string) bool { return true } // the format, with its entity-type letter, is the check
