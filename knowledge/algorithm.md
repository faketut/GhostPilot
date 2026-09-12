# Algorithm patterns cheatsheet

Reference only — the output format comes from the algorithm prompt (UMPIR, one
fenced code block, no comments in code), which wins on any conflict.

## Pattern triggers

| When the question says… | Reach for | Cost |
|---|---|---|
| sorted array, pair/triple summing to target | two pointers | O(n) |
| contiguous subarray/substring, longest/shortest under a constraint | sliding window | O(n) |
| sorted + find first/last/target | binary search | O(log n) |
| kth largest, top-k, merge k lists, streaming median | heap | O(n log k) |
| count/frequency, seen-before, group by key | hash map | O(n) |
| range sum, running total, equilibrium index | prefix sum | O(n) |
| overlap, merge, insert, meeting rooms | sort by start or end | O(n log n) |
| grid/graph, shortest path in steps, islands | BFS | O(V+E) |
| grid/graph, path enumeration, all solutions | DFS / backtracking | exponential |
| next greater/smaller, histogram, stock span | monotonic stack | O(n) |
| count ways, min/max cost, choose-or-skip | DP | O(n) / O(n·m) |
| connectivity, merge accounts, cycle detection | union-find | ~O(1) |
| prefix search, autocomplete, word list | trie | O(len) |

## Loop shapes

- two pointers: `l, r = 0, n-1`; move one pointer by comparison until `l >= r`
- window: `for right: add(right); while not ok(): remove(left); left += 1`;
  size is `right - left + 1`
- binary search: `lo, hi = 0, n`; `mid = (lo + hi) // 2`; `if pred(mid): hi = mid else: lo = mid + 1`
- BFS: `q = deque([start])` + `seen`, `for _ in range(len(q))` per level
- DP: `dp[i] = best over transitions i -> j`
- monotonic stack: `while stack and cmp(a[stack[-1]]): pop`, then push `i`

## Traps

- `while` (not `if`) in a window when one new element can break the invariant
  repeatedly.
- Binary search the *answer* (capacity, speed, days), not the input, when the
  input isn't sorted but feasibility is monotone.
- Recursion over 10⁵ elements overflows the stack — make it iterative.
- Python `//` floors toward −∞ (`-7 // 2 == -4`); C truncates toward zero.
