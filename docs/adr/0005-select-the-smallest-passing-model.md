# Select the smallest model that passes every gate

Model selection first rejects any candidate that fails the validation gate or a
deployment constraint. Among passing candidates, it prefers the smallest
TFLite artifact, then the smallest tensor arena, then the fewest
multiply-accumulate operations, and finally the highest validation macro-F1 as
a tie-breaker. This treats the validation thresholds as a floor and embedded
cost as the optimization objective without consulting the reserved test
partition.
