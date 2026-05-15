#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct AbstructAnimal {
    char name[64];
    int age;
} AbstructAnimal, *AbstructAnimalPtr;

typedef struct Student {
    AbstructAnimal animal;
    double score;
} Student, *StudentPtr;

StudentPtr create_student(char* name, int age, double score) {
    StudentPtr student = (StudentPtr)malloc(sizeof(Student));
    strcpy(student->animal.name, name);
    student->animal.age = age;
    student->score = score;
    return student;
}

void destroy_student(StudentPtr student) {
    free(student);
}

void print_student(StudentPtr student_ptr) {
    printf("Name: %s, Age: %d, Score: %.2f\n", student_ptr->animal.name, student_ptr->animal.age, student_ptr->score);
}

int main(int argc, char const* argv[]) {
    StudentPtr student = create_student("Alice", 18, 85.5);
    print_student(student);
    destroy_student(student);

    return 0;
}
