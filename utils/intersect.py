############################################################################
#
# To calculate the intersection point of two lines.
#
#
#
#
############################################################################


def intersect(line1: list[tuple], line2: list[tuple]):
    [(x1, y1), (x2, y2)] = line1
    [(x3, y3), (x4, y4)] = line2

    # Calculate slopes (m) and y-intercepts (b) for the two lines
    m1 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float("inf")  # Avoid division by zero
    b1 = y1 - m1 * x1 if m1 != float("inf") else None

    m2 = (y4 - y3) / (x4 - x3) if (x4 - x3) != 0 else float("inf")  # Avoid division by zero
    b2 = y3 - m2 * x3 if m2 != float("inf") else None

    # Check for parallel lines
    if m1 == m2:
        # Check for overlapping lines
        if b1 == b2:
            return True  # Lines overlap
        else:
            return False  # Lines are parallel but not overlapping
    else:
        # Check for intersection point
        x_intersect = (b2 - b1) / (m1 - m2) if m1 != float("inf") else x1
        y_intersect = m1 * x_intersect + b1 if m1 != float("inf") else m2 * x_intersect + b2

        # Check if the intersection point lies within the line segments
        if (
            min(x1, x2) <= x_intersect <= max(x1, x2)
            and min(y1, y2) <= y_intersect <= max(y1, y2)
            and min(x3, x4) <= x_intersect <= max(x3, x4)
            and min(y3, y4) <= y_intersect <= max(y3, y4)
        ):
            return True  # Lines intersect
        else:
            return False  # Lines are not parallel but do not intersect