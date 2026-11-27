resource "aws_security_group" "hr_database" {
  name = "hr-database"
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_s3_bucket_acl" "hr_exports" {
  bucket = "hr-exports-personal-data"
  acl    = "public-read"
}
