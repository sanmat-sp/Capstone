CREATE DATABASE capstone;

CREATE TABLE blind(
    img_id INT PRIMARY KEY,
    img LONGBLOB,
    block_no INT
);

CREATE TABLE non_blind(
    img_id INT PRIMARY KEY,
    img LONGBLOB,
    ll LONGBLOB,
    block_no INT
);