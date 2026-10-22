package redact

import (
	"compress/gzip"
	"io"
	"os"
	"regexp"
	"slices"
	"strings"
	"testing"
)

func TestEachKindIsReplacedByItsPlaceholder(t *testing.T) {
	r := New()
	cases := map[string]string{
		"Write to jane.doe+ir@example.co.uk today":         "Write to [EMAIL] today",
		"Call (425) 882-8080 or 425-882-8080":              "Call [PHONE] or [PHONE]",
		"or 425.882.8080, toll-free 1-800-642-7676":        "or [PHONE], toll-free [PHONE]",
		"London +44 20 7946 0958, Mumbai +91 98765 43210":  "London [PHONE], Mumbai [PHONE]",
		"US +1 (408) 996-1010.":                            "US [PHONE].",
		"Card 4111 1111 1111 1111 expires":                 "Card [CARD] expires",
		"card 4111-1111-1111-1111, amex 3782 822463 10005": "card [CARD], amex [CARD]",
		"plain 4012888888881881 too":                       "plain [CARD] too",
		"SSN 123-45-6789.":                                 "SSN [SSN].",
		"PAN ABCPE1234F filed":                             "PAN [PAN] filed",
		"Aadhaar 2341 2341 2346 or 234123412346":           "Aadhaar [AADHAAR] or [AADHAAR]",
	}
	for in, want := range cases {
		if got, _ := r.Text(in); got != want {
			t.Errorf("%q\n got %q\nwant %q", in, got, want)
		}
	}
}

func TestCountsAreByKindOnly(t *testing.T) {
	_, counts := New().Text("a@b.io, c@d.io and 123-45-6789")
	if counts[Email] != 2 || counts[SSN] != 1 || len(counts) != 2 {
		t.Fatalf("counts %v", counts)
	}
	if _, counts := New().Text("Total revenue was $245,122 million."); counts != nil {
		t.Fatalf("counts %v for a sentence with no personal data", counts)
	}
}

func TestLookalikesAreLeftAlone(t *testing.T) {
	r := New()
	keep := []string{
		// checksums
		"4111 1111 1111 1112", // fails Luhn
		"2341 2341 2341",      // fails Verhoeff
		"000-12-3456",         // no SSN has area 000
		"900-12-3456",         // nor 9xx
		"123-00-4567",         // nor group 00
		// figures, years and filing references
		"245,122 211,915 198,270",
		"2024 2023 2022 2021", // a row of years: 16 digits in fours
		"2025 2024 2023",      // 12 digits in fours
		"Fiscal Year 2024 2023 2022 2021 2020",
		"Commission File Number 001-37845",
		"Accession 0000950170-24-087843",
		"CIK 0000789019",
		"$9.68 and $11.80 per share",
		"(1,234) 5,678 (9,012)",
		"7,469 7,540 7,557 shares",
		"57,582 50,901 4,411.5",
		"(msft.htm, p.56) and (apple-sec-10k.htm, p. 31)",
		"10-K 10-Q 8-K",
		"June 30, 2024 2023-06-30 2024.06.30",
		"+1.5% +2.3 points",
		"12.5% 3.1415926535",
		// slices of longer digit runs are never taken
		"1 2024 2023 2022 2021",
		"94111 1111 1111 1111",
		"4111 1111 1111 1111 2222 3333",
		"ref 425-882-80801",
		// not personal data
		"ABCDE1234F", // PAN's fourth letter must be an entity type
		"user@localhost",
		"sales@ 2024",
	}
	for _, s := range keep {
		if got, counts := r.Text(s); got != s {
			t.Errorf("%q became %q (%v)", s, got, counts)
		}
	}
}

func TestOnlyTheKindsAskedForAreRedacted(t *testing.T) {
	if got, _ := New(Email).Text("a@b.io 123-45-6789"); got != "[EMAIL] 123-45-6789" {
		t.Fatalf("got %q", got)
	}
}

// Everything that, changed, would change an answer: the figures the answer
// check reads (rag.py _ANSWER_FIGURE), years, EPS, and citations.
var protected = []*regexp.Regexp{
	regexp.MustCompile(`\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+`),    // figures, share counts
	regexp.MustCompile(`(?:^|[^0-9])((?:19|20)\d{2})(?:[^0-9]|$)`), // years
	regexp.MustCompile(`\$\s?\(?\d+\.\d{2}\)?`),                    // EPS
	regexp.MustCompile(`\([^()]+?\.(?:pdf|html?),\s*p\.?\s*\d+\)`), // citations
	regexp.MustCompile(`(?i)(?:page|p\.)\s*\d+`),                   // page references
	regexp.MustCompile(`\d+(?:\.\d+)?\s?%`),                        // percentages
}

func tokens(s string) []string {
	var out []string
	for _, re := range protected {
		out = append(out, re.FindAllString(s, -1)...)
	}
	return out
}

func readFixture(t *testing.T, name string) string {
	t.Helper()
	f, err := os.Open("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var r io.Reader = f
	if strings.HasSuffix(name, ".gz") {
		gz, err := gzip.NewReader(f)
		if err != nil {
			t.Fatal(err)
		}
		r = gz
	}
	raw, err := io.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

// The required test: the redactor over both sample 10-Ks (their text as the
// app sends it) and every benchmark question. Not one figure, year, EPS value,
// share count or citation may change; every benchmark question must come out
// untouched; and in the filings, only the companies' own published phone
// numbers may be found (each one listed below, reviewed by hand).
func TestNoFigureYearOrCitationInTheSamplesOrQuestionsChanges(t *testing.T) {
	r := New()
	expected := map[string][]string{
		// The cover pages' and signature pages' contact numbers: real phone
		// numbers, correctly found, and no figure among their digits
		"microsoft-10k.txt.gz": {"(425) 882-8080"},
		"apple-10k.txt.gz":     {"(408) 996-1010"},
	}
	for fixture, allowed := range expected {
		text := readFixture(t, fixture)
		redacted, counts := r.Text(text)
		if !slices.Equal(tokens(text), tokens(redacted)) {
			t.Errorf("%s: redaction changed a figure, year, EPS value, share count or citation", fixture)
		}
		for _, m := range r.Find(text) {
			if !slices.Contains(allowed, m.Value) {
				t.Errorf("%s: unexpected %s %q (false positive?)", fixture, m.Kind, m.Value)
			}
		}
		t.Logf("%s: %d characters, redacted %v", fixture, len(text), counts)
	}

	questions := readFixture(t, "eval-questions.txt")
	lines := strings.Split(strings.TrimSpace(questions), "\n")
	if len(lines) < 50 {
		t.Fatalf("only %d benchmark texts in the fixture", len(lines))
	}
	for _, q := range lines {
		if got, counts := r.Text(q); got != q {
			t.Errorf("benchmark question changed (%v): %q", counts, q)
		}
	}
}
