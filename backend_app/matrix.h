#pragma once

#include <vector>

template<typename T>
class Matrix
{
private:
	std::vector<T> data;
	const size_t n_rows;
	const size_t n_cols;

	//std::reference_wrapper<T> operator()(const size_t& row, const size_t& column) const {
	//	return std::reference_wrapper<T>(data[row * n_cols + column]);
	//};

public:

	Matrix(const size_t& n_rows, const size_t& n_cols) : n_rows(n_rows), n_cols(n_cols), data(n_rows* n_cols) {};

	size_t getNumberOfRows() const { return n_rows; }
	size_t getNumberOfColumns() const { return n_cols; }

	void fillData(const size_t& row, const size_t& column, const T& new_data) {
		data[row * n_cols + column] = new_data;
	};

	std::vector<T> getRow(const size_t& i) const {
		if (i >= n_rows)
			throw std::runtime_error("The requested row in matrix does not exist!");

		std::vector<T> result;
		result.reserve(n_cols);
		for (size_t c = 0; c < n_cols; ++c) {
			result.push_back(data[i * n_cols + c]);
		}
		return result;
	}

	std::vector<T> getColumn(const size_t& i) const {
		if (i >= n_cols)
			throw std::runtime_error("The requested column in matrix does not exist!");

		std::vector<T> result;
		result.reserve(n_rows);
		for (size_t r = 0; r < n_rows; ++r) {
			result.push_back(data[r * n_cols + i]);
		}
		return result;
	}

	std::vector<T> operator[](const size_t& i) const {
		return getRow(i);
	}
};