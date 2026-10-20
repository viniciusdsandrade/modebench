"""The one to one match of hypothesis speakers to reference speakers.

The labels of the application and those of the gold set are different names
for the same people, so each measure first finds the match that agrees most:
the match with the largest total of a table of agreement, by the Hungarian
algorithm. A speaker that is left over has no match.
"""

from collections.abc import Mapping

INFINITY = float("inf")


def _cheapest_square(cost: list[list[float]]) -> list[int]:
    """Return, for each row of a square table, the column of the cheapest perfect match.

    This is the shortest augmenting path form of the Hungarian algorithm, with
    potentials, in O(n^3).
    """
    size = len(cost)
    row_potential = [0.0] * (size + 1)
    column_potential = [0.0] * (size + 1)
    row_of_column = [0] * (size + 1)
    previous = [0] * (size + 1)
    for row in range(1, size + 1):
        row_of_column[0] = row
        column = 0
        least = [INFINITY] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column] = True
            current = row_of_column[column]
            delta = INFINITY
            chosen = 0
            for other in range(1, size + 1):
                if used[other]:
                    continue
                reduced = (
                    cost[current - 1][other - 1] - row_potential[current] - column_potential[other]
                )
                if reduced < least[other]:
                    least[other] = reduced
                    previous[other] = column
                if least[other] < delta:
                    delta = least[other]
                    chosen = other
            for other in range(size + 1):
                if used[other]:
                    row_potential[row_of_column[other]] += delta
                    column_potential[other] -= delta
                else:
                    least[other] -= delta
            column = chosen
            if row_of_column[column] == 0:
                break
        while column:
            back = previous[column]
            row_of_column[column] = row_of_column[back]
            column = back
    column_of_row = [0] * size
    for column in range(1, size + 1):
        if row_of_column[column]:
            column_of_row[row_of_column[column] - 1] = column - 1
    return column_of_row


def best_mapping(table: Mapping[str, Mapping[str, float]]) -> dict[str, str]:
    """Return the match of rows to columns with the largest total agreement.

    `table[row][column]` is how much the row speaker and the column speaker
    agree, for example the seconds or the words they share. A pair that does
    not agree at all is never matched.
    """
    rows = sorted(table)
    columns = sorted({column for row in table.values() for column in row})
    if not rows or not columns:
        return {}
    size = max(len(rows), len(columns))
    weights = [
        [
            table[rows[row]].get(columns[column], 0.0)
            if row < len(rows) and column < len(columns)
            else 0.0
            for column in range(size)
        ]
        for row in range(size)
    ]
    largest = max(value for line in weights for value in line)
    cost = [[largest - value for value in line] for line in weights]
    chosen = _cheapest_square(cost)
    mapping: dict[str, str] = {}
    for row, column in enumerate(chosen):
        if row < len(rows) and column < len(columns) and weights[row][column] > 0:
            mapping[rows[row]] = columns[column]
    return mapping


def mapped_total(table: Mapping[str, Mapping[str, float]], mapping: Mapping[str, str]) -> float:
    """Return the agreement that a match keeps."""
    return sum(table[row].get(column, 0.0) for row, column in mapping.items())
