#pragma once

#include <vector>

template<typename T>
class Matrix
{
private:
	std::vector<T> data;
	const size_t n_rows;
	const size_t n_cols;

public:

	Matrix(const size_t& n_rows, const size_t& n_cols) : n_rows(n_rows), n_cols(n_cols), data(n_rows* n_cols) {};

	size_t getNumberOfRows() const { return n_rows; }
	size_t getNumberOfColumns() const { return n_cols; }

	void fillData(const size_t& row, const size_t& column, const T& new_data) {
		data[row * n_cols + column] = new_data;
	};

	std::vector<std::reference_wrapper<const T>> getRow(size_t r) const {
		auto start = std::begin(data) + r * n_cols;
		return{ start, start + n_cols };
	}

	std::vector<std::reference_wrapper<const T>> getColumn(size_t c) const {
		std::vector<std::reference_wrapper<const T>> result;
		result.reserve(n_rows);
		for (size_t r = 0; r < n_rows; ++r) {
			result.push_back(std::reference_wrapper<const T>(data[r * n_cols + c]));
		}
		return result;
	}

	std::vector<std::reference_wrapper<const T>> operator[](const size_t& i) const {
		return getRow(i);
	}
};