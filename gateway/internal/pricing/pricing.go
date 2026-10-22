// Package pricing turns token counts into dollars and into budget units.
//
// Budgets count weighted tokens: one unit is what one gpt-4o-mini input token
// costs ($0.15 per million), so a gpt-4o-mini output token is 4 units, a cached
// input token 0.5, and an embedding token about 0.13. A budget is then one
// number however a consumer's calls mix models and kinds of token.
package pricing

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/dasosis/financial-rag/gateway/internal/usage"
)

// Price is a model's list price, USD per million tokens.
type Price struct {
	Input  float64 `json:"input"`
	Cached float64 `json:"cached"` // cached input; 0 means the input price
	Output float64 `json:"output"`
}

// Reference is the price of one budget unit: a gpt-4o-mini input token.
const Reference = 0.15 / 1e6

// Defaults are list prices for the models the app uses (and a few it may be
// switched to). Matched by prefix, longest first, so "gpt-4o-mini-2024-07-18"
// is priced as "gpt-4o-mini". OpenRouter names models "openai/…".
var Defaults = map[string]Price{
	"gpt-4o-mini":                   {Input: 0.15, Cached: 0.075, Output: 0.60},
	"openai/gpt-4o-mini":            {Input: 0.15, Cached: 0.075, Output: 0.60},
	"gpt-4o":                        {Input: 2.50, Cached: 1.25, Output: 10.00},
	"gpt-4.1-mini":                  {Input: 0.40, Cached: 0.10, Output: 1.60},
	"gpt-5-mini":                    {Input: 0.25, Cached: 0.025, Output: 2.00},
	"text-embedding-3-small":        {Input: 0.02},
	"openai/text-embedding-3-small": {Input: 0.02},
}

// Unknown prices a model missing from the table: as gpt-4o, so an unpriced
// model is charged too much rather than too little.
var Unknown = Price{Input: 2.50, Cached: 1.25, Output: 10.00}

// Table looks up models' prices.
type Table struct {
	prices map[string]Price
}

// New is the defaults with `overrides` (JSON {"model": {"input": …}}) on top.
func New(overrides string) (*Table, error) {
	t := &Table{prices: map[string]Price{}}
	for name, p := range Defaults {
		t.prices[name] = p
	}
	if strings.TrimSpace(overrides) == "" {
		return t, nil
	}
	var extra map[string]Price
	if err := json.Unmarshal([]byte(overrides), &extra); err != nil {
		return nil, fmt.Errorf("GATEWAY_PRICES: %w", err)
	}
	for name, p := range extra {
		t.prices[name] = p
	}
	return t, nil
}

// Lookup is a model's price, and whether the table has it.
func (t *Table) Lookup(model string) (Price, bool) {
	best, found := "", false
	for name := range t.prices {
		if strings.HasPrefix(model, name) && len(name) > len(best) {
			best, found = name, true
		}
	}
	if !found {
		return Unknown, false
	}
	return t.prices[best], true
}

// Cost is what a call's usage cost, in dollars.
func (t *Table) Cost(model string, u usage.Usage) float64 {
	p, _ := t.Lookup(model)
	cachedPrice := p.Cached
	if cachedPrice == 0 {
		cachedPrice = p.Input
	}
	cached := min(u.Cached(), u.PromptTokens)
	return (float64(u.PromptTokens-cached)*p.Input +
		float64(cached)*cachedPrice +
		float64(u.CompletionTokens)*p.Output) / 1e6
}

// Units is a call's usage in budget units, rounded up.
func (t *Table) Units(model string, u usage.Usage) int64 {
	return ToUnits(t.Cost(model, u))
}

// ToUnits converts dollars to budget units, rounded up.
func ToUnits(usd float64) int64 {
	return int64(math.Ceil(usd/Reference - 1e-9))
}
