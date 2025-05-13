from __future__ import annotations

import csv
from copy import copy, deepcopy

_align = {
    "l": "<",
    "r": ">",
    "c": "^",
}


class Table:
    """ A simple table class for printing data in tabular format.

    +---------------------------------+
    |             Title               |
    +---------------------------------+
    | Header1      Header2     ...    |
    |=================================|
    | Cell(1,1)    Cell(1,2)   ...    |
    | Cell(2,1)    Cell(2,2)   ...    |
    | ...           ...        ...    |
    +---------------------------------+

    Attributes:
        title: title of the table
        headers: column headers
        rows: list of rows, each row is a list of cells
        align: alignment of each column,
            "l" for left, "r" for right and "c" for center.
            If not specified, all columns are left-aligned.
        col_spacing: spacing between columns
        margin: spacing between the table and vertical borders
        unit: unit of each column
    """

    def __init__(
        self,
        title: str,
        headers: list[str],
        align: str = "",
        col_spacing: int = 5,
        margin: int = 1,
        unit: list[str] | None = None,
    ):
        self.title = title
        self.headers = headers
        self.rows = []
        self.align = (align or "l" * len(headers)).lower()
        self.col_spacing = col_spacing
        self.margin = margin
        self.unit = unit or [""] * len(headers)
        assert set(self.align) <= {"l", "r", "c"}
        assert len(self.align) == len(self.headers)
        assert len(self.unit) == len(self.headers)

    def copy(self, rows: list[int] | None = None) -> Table:
        table = Table(self.title, copy(self.headers), self.align,
                      self.col_spacing, self.margin, copy(self.unit))
        if rows is None:
            table.rows = deepcopy(self.rows)
        else:
            table.rows = [self.rows[i] for i in rows] if rows else []
        return table

    def add_row(self, row: list[str]) -> Table:
        assert len(row) == len(self.headers)
        self.rows.append(row)
        return self

    def print(self, file=None) -> None:
        if file is None:
            print(str(self))
        else:
            print(str(self), file=file)

    def __str__(self):
        headers = self.headers or ["[Empty Table]"]
        rows = self.rows or [[""] * len(headers)]
        units = self.unit or [""] * len(headers)
        aligns = [_align[a] for a in self.align] or ["^"]

        max_widths = [len(h) for h in headers]
        for row in self.rows:
            for i, cell in enumerate(row):
                max_widths[i] = max(max_widths[i], len(str(cell) + units[i]))

        full_width = sum(max_widths) + self.col_spacing * (len(headers) - 1)
        # assert len(
        #     self.title) <= full_width, f"title is too long: {self.title}"
        while len(self.title) > full_width:
            # increase column spacing
            self.col_spacing += 1
            full_width = sum(max_widths) + self.col_spacing * (len(headers) -
                                                               1)

        col_sp = " " * self.col_spacing

        lb = "|" + " " * self.margin  # left border
        rb = " " * self.margin + "|"  # right border
        tbb = "+" + "-" * (full_width +
                           2 * self.margin) + "+"  # top and bottom border
        sep = "|" + "=" * (full_width + 2 * self.margin
                           ) + "|"  # separator between header and table

        t_rows = []
        for row in [headers] + rows:
            header = row is headers  # do not add unit to header
            t_rows.append(lb + col_sp.join(
                f"{str(cell) + ('' if header else unit):{align}{width}}"
                for cell, width, align, unit in zip(row, max_widths, aligns,
                                                    units)) + rb)

        title = lb + self.title.center(full_width) + rb

        return "\n".join([tbb, title, tbb, t_rows[0], sep] + t_rows[1:] +
                         [tbb])

    def __repr__(self):
        return f"<Table {self.title}>"

    def __len__(self):
        return len(self.rows)

    def write(self, filename: str):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
            writer.writerows(self.rows)


__all__ = ["Table"]

if __name__ == "__main__":
    tbl = Table("Test Table", ["Header1", "Header2", "Header3"], align="lcr")
    tbl.add_row(["Cell(1,1)", "Cell(1,2)", "Cell(1,3)"])
    tbl.add_row(["Cell(2,1)", "Cell(2,2)", "Cell(2,3)"])
    tbl.add_row(["...", "", ""])
    tbl.print()
